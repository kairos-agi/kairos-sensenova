import math
import torch
import torch.distributed as dist
from einops import rearrange
from torch import nn

def broadcast_split_tensor(tensor, seq_dim=0, process_group=None):
    rank = dist.get_rank(group=process_group)
    world_size = dist.get_world_size(group=process_group)
    chunks = torch.chunk(tensor, world_size, dim=seq_dim)
    return chunks[rank].contiguous()

import torch.distributed as dist

class ParallelVAEWrapper(nn.Module):
    def __init__(
        self, 
        original_vae, 
        context_parallel_group, 
        parallel_mode="cp",
        time_compression_ratio=4, 
        overlap_size=4,
    ):
        super().__init__()
        self.vae = original_vae
        self.parallel_group = context_parallel_group
        self.world_size = dist.get_world_size(group=context_parallel_group)
        self.rank = dist.get_rank(group=context_parallel_group)
        self.parallel_mode = parallel_mode
        
        self.time_scale = time_compression_ratio
        self.overlap_size = overlap_size

    def _prepare_inputs_with_halo(self, latents: torch.Tensor):
        b, c, t, h, w = latents.shape
        # Replicate Padding
        if t % self.world_size != 0:
            target_t = math.ceil(t / self.world_size) * self.world_size
            pad_len = target_t - t
            # Copy the last frame pad_len times
            if t > pad_len:
                pad_part = latents[:, :, -(pad_len+1):-1, :, :]
                pad_tensor = torch.flip(pad_part, dims=[2])
            else:
                # If the video is too short to reflect, then downgraded to a copy.
                last_frame = latents[:, :, -1:, :, :]
                pad_tensor = last_frame.repeat(1, 1, pad_len, 1, 1)
            latents_padded = torch.cat([latents, pad_tensor], dim=2)
        else:
            pad_len = 0
            latents_padded = latents

        chunk_size = latents_padded.shape[2] // self.world_size
        start_idx = self.rank * chunk_size
        end_idx = (self.rank + 1) * chunk_size

        halo_start_idx = start_idx - self.overlap_size
        if self.rank == 0:
            chunk_part = latents_padded[:, :, start_idx:end_idx, :, :]
            tail_pad_len = self.overlap_size
            if chunk_part.shape[2] > tail_pad_len:
                pad_source = chunk_part[:, :, -tail_pad_len:, :, :]
                tail_halo = torch.flip(pad_source, dims=[2])
            else:
                last_frame = chunk_part[:, :, -1:, :, :]
                tail_halo = last_frame.repeat(1, 1, tail_pad_len, 1, 1)
                
            local_latents = torch.cat([chunk_part, tail_halo], dim=2)
            pad_mode = "tail"
        elif halo_start_idx < 0:
            # Extract the valid real part [0, end_idx]
            real_part = latents_padded[:, :, 0:end_idx, :, :]
            # Missing length
            missing_len = -halo_start_idx
            
            if latents_padded.shape[2] > missing_len:
                reflect_part = latents_padded[:, :, 1:missing_len+1, :, :]
                front_pad = torch.flip(reflect_part, dims=[2])
            else:
                first_frame = latents_padded[:, :, 0:1, :, :]
                front_pad = first_frame.repeat(1, 1, missing_len, 1, 1)
                
            local_latents = torch.cat([front_pad, real_part], dim=2)
            pad_mode = "head"
        else:
            local_latents = latents_padded[:, :, halo_start_idx:end_idx, :, :].contiguous()
            pad_mode = "head"

        meta = {
            "original_t": t,
            "chunk_size": chunk_size,
            "pad_mode": pad_mode,
            "pad_len": pad_len 
        }
        
        return local_latents, meta

    def _post_process_and_gather(self, local_output: torch.Tensor, meta: dict):
        if local_output.device.type == 'cpu':
            mv_device = f"cuda:{self.rank % torch.cuda.device_count()}"
            local_output = local_output.to(mv_device)

        scale = self.time_scale
        anchor_len = 1 * scale
        if meta["pad_mode"] == "head":
            # Rank > 0: Cut off Overlap-1, keep 1 unit as an Anchor (for merging)
            cut_len = (self.overlap_size - 1) * scale
            local_valid = local_output[:, :, cut_len:, :, :]
            
        elif meta["pad_mode"] == "tail":
            # Rank 0: Cut off all unnecessary padding at the tail.
            tail_cut_len = self.overlap_size * scale
            local_valid = local_output[:, :, :-tail_cut_len, :, :]
            
            # Rank 0 currently has a length of Chunk_Pix
            # But Rank > 0 has a length of Chunk_Pix + Anchor_Pix
            # To achieve Gathering, Rank 0 needs to be padded with a dummy data segment.
            dummy_shape = list(local_valid.shape)
            dummy_shape[2] = anchor_len
            dummy = torch.zeros(dummy_shape, dtype=local_valid.dtype, device=local_valid.device)
            local_valid = torch.cat([local_valid, dummy], dim=2)

        local_p = rearrange(local_valid, "B C T H W -> T B C H W").contiguous()
        out_list = [torch.empty_like(local_p) for _ in range(self.world_size)]
        dist.all_gather(out_list, local_p, group=self.parallel_group)

        processed_list = []
        for i, tensor in enumerate(out_list):
            t_chunk = rearrange(tensor, "T B C H W -> B C T H W")
            
            if i == 0:
                # Remove Dummy Anchor
                t_chunk_clean = t_chunk[:, :, :-anchor_len, :, :]
                processed_list.append(t_chunk_clean)
            else: 
                prev_chunk = processed_list[-1]
                # Retrieve the last frame of the previous block (Frame N)
                prev_last_frame = prev_chunk[:, :, -1:, :, :] 
                # Delete the last frame of the previous block
                processed_list[-1] = prev_chunk[:, :, :-1, :, :] 
                # Retrieve the first frame of the current block (Frame N generated by Halo)
                curr_first_frame = t_chunk[:, :, 0:1, :, :]
                # Linear fusion (0.5 * Rank_i-1 + 0.5 * Rank_i)
                blended_frame = (prev_last_frame + curr_first_frame) / 2.0
                # [Blended_Frame, Rest_of_Chunk]
                t_chunk_blended = torch.cat([blended_frame, t_chunk[:, :, 1:, :, :]], dim=2)
                processed_list.append(t_chunk_blended)

        output_full = torch.cat(processed_list, dim=2)
        
        original_in_t = meta["original_t"]
        valid_total_t = (original_in_t - 1) * scale + 1
        
        if valid_total_t > output_full.shape[2]:
             valid_total_t = output_full.shape[2]

        final_output = output_full[:, :, :valid_total_t, :, :]
        
        if self.rank == 0:
             print(f"[CP VAE] Output T: {final_output.shape[2]} (Target: {valid_total_t})")
             pass

        return final_output

    def decode(self, latents: torch.Tensor, device, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        if self.rank == 0:
            print(f"[CP VAE] Paralle with mode T: {self.parallel_mode}")
        if self.parallel_mode == "cp" :
            return self.decode_cp(latents, device)
        
        return self.decode_sp(latents, device, tiled, tile_size, tile_stride)
    
    def decode_cp(self, latents: torch.Tensor,  device, **kwargs):
        local_latents, meta = self._prepare_inputs_with_halo(latents)
        decode_out = self.vae.decode(local_latents, device, False)
        final_output = self._post_process_and_gather(decode_out, meta)
        return final_output
    
    def decode_sp(self, latents: torch.Tensor, device, tiled, tile_size, tile_stride):
        height, width = latents.shape[2], latents.shape[3]
        using_sp = True
        if width % self.world_size == 0:
            split_dim = 3
        elif height % self.world_size == 0:
            split_dim = 2
        else : 
            using_sp = False
        if using_sp:
            images = self.decode_dist(latents, self.world_size, self.rank , split_dim)
        else :
            print(f"Fall back to naive decode mode")
            images = self.vae.decode(latents, device, tiled, tile_size, tile_stride)
        return images
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vae, name)
        
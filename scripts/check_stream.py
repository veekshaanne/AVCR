from src.cvpipe.stream import iter_frame_chunks

total, sizes = 0, []
for idx, frames in iter_frame_chunks("data/raw/cuts.mp4", chunk_size=16):
    total += len(idx)
    sizes.append(len(idx))
print("total frames :", total)
print("chunk sizes  :", sizes)
print("last chunk   :", idx[0], "to", idx[-1], "| shape", frames.shape)

# read only scene 1 (frames 72..143)
n = sum(len(i) for i, _ in iter_frame_chunks("data/raw/cuts.mp4", 16, start=72, end=144))
print("scene 1 frames:", n)

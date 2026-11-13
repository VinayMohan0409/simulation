"""SAM 3 text-prompted instance segmentation as a worker process, run with the SAM 3 environment's Python (3.12; the lawm environments are 3.10), so it imports nothing of lawm.

Protocol (stdin -> stdout, binary): each message is an 8-byte little-endian length followed by a pickle.
Request:  dict(images=[uint8 (H, W, 3), ...], prompts=[str, ...], threshold=float).
Reply:    [per image: {prompt: dict(masks=bool (n, H, W), scores=float32 (n,), boxes=float32 (n, 4) x0 y0 x1 y1)}] or dict(error=str).
A request of None ends the worker. Everything the model prints goes to stderr, so stdout carries only replies."""
import pickle
import struct
import sys
import traceback


def read_msg(f):
    head = f.read(8)
    if len(head) < 8:
        return None
    n = struct.unpack("<Q", head)[0]
    if n == 0:  # Segmenter.close() sends a zero length: a clean exit
        return None
    return pickle.loads(f.read(n))


def write_msg(f, obj):
    data = pickle.dumps(obj, protocol=4)
    f.write(struct.pack("<Q", len(data)))
    f.write(data)
    f.flush()


def main():
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(enable_inst_interactivity=True)  # text prompts and SAM 2-style point prompts (predict_inst)
    processor = Sam3Processor(model)
    write_msg(out, dict(ready=True, torch=torch.__version__))
    while True:
        req = read_msg(sys.stdin.buffer)
        if req is None:
            break
        try:
            processor.set_confidence_threshold(float(req.get("threshold", 0.3)))
            reply = []
            for img in req["images"]:
                # fp16 autocast + flash / memory-efficient attention on every GPU (same numerics everywhere). Plain fp32 fails (SAM 3 casts parts to bf16 itself);
                # bf16 on the 2080 Ti falls back to the math kernel (9.0 GB peak, OOM next to the sim) vs 5.4 GB here (measured 2026-11-13).
                with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]), torch.autocast("cuda", dtype=torch.float16):
                    state = processor.set_image(Image.fromarray(img))
                    if "points" in req:  # point prompts: one (coords (n, 2) x y pixels, labels (n,) 1/0) per image, or None; reply: the best of 3 masks
                        pr = req["points"][len(reply)]
                        if pr is None:
                            reply.append(None)
                            continue
                        masks, scores, _ = model.predict_inst(state, point_coords=np.asarray(pr[0], np.float32), point_labels=np.asarray(pr[1], np.int32), multimask_output=True)
                        b = int(np.argmax(scores))
                        reply.append(dict(mask=masks[b] > 0, score=float(scores[b]), masks=masks > 0, scores=np.asarray(scores, np.float32)))  # best + all 3
                        continue
                    per = {}
                    for prompt in req["prompts"]:
                        processor.reset_all_prompts(state)
                        o = processor.set_text_prompt(state=state, prompt=prompt)
                        per[prompt] = dict(masks=o["masks"][:, 0].cpu().numpy().astype(bool), scores=o["scores"].float().cpu().numpy(),
                                           boxes=o["boxes"].float().cpu().numpy())
                reply.append(per)
            write_msg(out, reply)
        except Exception:
            write_msg(out, dict(error=traceback.format_exc()))


if __name__ == "__main__":
    main()

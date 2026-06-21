# Hand-Drawn Architecture Diagram — Drawing Blueprint

> For sir's point #12. This is a **drawing guide**, not the final figure. Copy it onto paper by hand.
> Every box label and dimension below is taken from the verified `docs/model_spec.md`
> (read from the instantiated model, not guessed). Draw boxes left-to-right / top-to-bottom as a U.

---

## Page layout (draw the U shape)

Lay the page **landscape**. Think of a big letter **U**:
- Left arm going **down** = ENCODER (image shrinks, channels grow)
- Bottom of the U = BOTTLENECK (latent space)
- Right arm going **up** = DECODER (image grows back)
- A small **side branch on the far left** = SCATTERING (feeds into every level via fusion)
- Two small boxes on the right = the two OUTPUT HEADS

---

## 1. INPUT (top-left)

```
┌─────────────────────────┐
│  PAUT amplitude patch    │
│      1 × 256 × 256        │
└─────────────────────────┘
```
One arrow down into enc1.

---

## 2. ENCODER — left arm, going DOWN (4 boxes)

Draw 4 stacked boxes. Each box = "2× (Conv3×3 → GroupNorm → ReLU)".
Between each box draw a small **↓ MaxPool 2** arrow.

| Box label | Write inside the box (C × H × W) |
|-----------|----------------------------------|
| **enc1**  | 32 × 256 × 256 |
| ↓ MaxPool | |
| **enc2**  | 64 × 128 × 128 |
| ↓ MaxPool | |
| **enc3**  | 128 × 64 × 64 |
| ↓ MaxPool | |
| **enc4**  | 256 × 32 × 32 |
| ↓ MaxPool | |

(Channels double 32→64→128→256; size halves 256→128→64→32.)

---

## 3. BOTTLENECK — bottom of the U (latent space)

```
┌──────────────────────────────┐
│        BOTTLENECK             │
│        512 × 16 × 16          │
│  (= latent space, 131,072 act)│
└──────────────────────────────┘
```

From the bottleneck draw a small arrow to a tiny box:
```
 Global Avg Pool → latent vector (512)
```
This 512 vector goes to the classifier (see head 2).

---

## 4. SCATTERING BRANCH — far-left side branch (fixed, not trained)

Draw this separately on the far left, with a **dashed box** to show it is FROZEN.

```
╔══════════════════════════════╗   ← dashed = fixed / not trained
║  Wavelet Scattering          ║
║  J=2, L=8, order=2           ║
║  output: 81 × 64 × 64        ║
║  (81 coeffs, 0 params)       ║
╚══════════════════════════════╝
```

From it, draw arrows into **5 small fusion boxes** (one per encoder level + bottleneck):

```
 fuse1 → fuse2 → fuse3 → fuse4 → fuseb
```
Label these "SCN-Fusion + CBAM (1×1 conv adapts 81→level width)".
Each fuse box sits beside its matching enc box (fuse1↔enc1, … fuseb↔bottleneck).
Draw a short arrow from each fuse box INTO the encoder box at the same level.

> Caption note to write under this branch:
> "Scattering filters are fixed buffers (0 trainable params) — the SCN physics prior.
> Only the small 1×1 fusion convs are learned."

---

## 5. DECODER — right arm, going UP (4 boxes)

Draw 4 stacked boxes going up. Between each, an **↑ Up-conv (ConvTranspose2d ×2)** arrow.
**Key feature:** each decoder box receives a SKIP arrow from the matching encoder box,
but the skip passes through an **Attention Gate** first. Draw the skip as a horizontal
dashed arrow from enc → small circle "AG" (Attention Gate) → decoder box.

| Box label | Write inside (C × H × W) | Skip from |
|-----------|--------------------------|-----------|
| **up4** | 256 × 32 × 32 | enc4 (via AG) |
| ↑ Up-conv | | |
| **up3** | 128 × 64 × 64 | enc3 (via AG) |
| ↑ Up-conv | | |
| **up2** | 64 × 128 × 128 | enc2 (via AG) |
| ↑ Up-conv | | |
| **up1** | 32 × 256 × 256 | enc1 (via AG) |

(Mirror image of encoder: channels halve, size doubles back to 256.)

---

## 6. THE TWO OUTPUT HEADS (top-right)

### Head 1 — Segmentation (from up1)
```
┌─────────────────────────────┐
│  seg_head  Conv2d(32→3, 1×1) │
│  output: 3 × 256 × 256       │
│  (background / porosity / slag)│
└─────────────────────────────┘
```
Arrow from up1 → seg_head.

### Head 2 — Classification (from latent + scattering)
```
   latent (512) ─┐
                 ├─► concat (593) → Linear(128) → ReLU → Linear(2)
 scatter vec(81)─┘                                         │
                                                  output: 2 classes
                                                  (porosity / slag)
```
Draw the 512 latent arrow and an 81 global-scattering arrow joining into "concat 593".

---

## 7. Boxes to label as "fixed / frozen" (use dashed borders)
- Scattering branch (81 × 64 × 64) — **0 params**

Everything else is trainable. **Total trainable params: 8,798,151.**
Write that figure in the bottom corner of the page.

---

## Quick legend to draw in a corner

```
─────►  data flow              ↓ MaxPool 2 (downsample)
- - - ►  skip connection        ↑ Up-conv ×2 (upsample)
  AG    Attention Gate          ╔═╗ dashed = fixed (not trained)
 fuse   SCN-Fusion + CBAM
```

---

## One-line summary to write as the figure caption

> **SCN-Attention U-Net:** a fixed Wavelet-Scattering physics prior (81 coeffs) is fused
> (CBAM) into every level of an Attention U-Net (encoder 32→64→128→256, bottleneck 512×16×16).
> Two heads share the encoder: a 3-class segmentation head and a 2-class type classifier
> that also reads the 81-D scattering vector. 8.8 M trainable parameters; scattering filters add 0.

---

### Tips for drawing it cleanly
1. Pencil the **U outline first** (4 down, 1 bottom, 4 up), then fill dimensions.
2. Keep all enc/up boxes the **same size**; only the numbers inside change.
3. Use **one color** for trainable, **dashed/another color** for the frozen scattering branch.
4. Draw the 5 fuse boxes **small** along the left so they don't clutter the U.
5. Put the two heads on the **right edge**, clearly separated, so the "two-task" design is obvious.

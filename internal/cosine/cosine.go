// Package cosine provides cosine distance and best-match helpers for
// L2-normalized embedding vectors such as those produced by ArcFace.
//
// For L2-normalized vectors, cosine distance equals (1 - dot(a, b)).
// Range is [0, 2]: 0 = identical, 1 = orthogonal, 2 = anti-parallel.
// Typical face-match threshold is ~0.45 (similarity >= 0.55).
package cosine

// Distance returns the cosine distance between two slices.
//
// If the lengths differ or are zero, returns 1.0 ("no match" sentinel).
// Inputs are L2-normalized on output by the embedder, so naive 1 - dot
// is correct; no manual normalization is required here.
func Distance(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 1.0
	}
	var dot float32
	// 8-wide unroll helps auto-vectorization on hot paths.
	i := 0
	for ; i+8 <= len(a); i += 8 {
		dot += a[i]*b[i] +
			a[i+1]*b[i+1] +
			a[i+2]*b[i+2] +
			a[i+3]*b[i+3] +
			a[i+4]*b[i+4] +
			a[i+5]*b[i+5] +
			a[i+6]*b[i+6] +
			a[i+7]*b[i+7]
	}
	for ; i < len(a); i++ {
		dot += a[i] * b[i]
	}
	return 1.0 - dot
}

// BestMatch returns the minimum cosine distance from probe against any
// reference, plus the index of the winning reference (or -1 if refs empty).
func BestMatch(probe []float32, refs [][]float32) (float32, int) {
	if len(refs) == 0 {
		return 1.0, -1
	}
	best := float32(1.0)
	bestIdx := -1
	for i, r := range refs {
		if d := Distance(probe, r); d < best {
			best = d
			bestIdx = i
		}
	}
	return best, bestIdx
}

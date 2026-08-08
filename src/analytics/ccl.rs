//! Connected-component labeling on binary masks with 8-connectivity, pixel area, and bounds.

/// One 8-connected foreground region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Component {
    /// Foreground pixel count.
    pub area: u32,
    pub min_x: u32,
    pub min_y: u32,
    /// Inclusive.
    pub max_x: u32,
    /// Inclusive.
    pub max_y: u32,
}

/// Reusable two-pass union-find labeler.
#[derive(Default)]
pub struct ConnectedComponents {
    /// Per-pixel label: 0 = background, else component index + 1 (after
    /// `label()` returns).
    labels: Vec<u32>,
    parent: Vec<u32>,
    comp_of_root: Vec<u32>,
}

impl ConnectedComponents {
    pub fn new() -> Self {
        Self::default()
    }

    /// Per-pixel labels from the last `label()` call: 0 for background,
    /// `i + 1` for pixels of `components[i]`.
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// Label all 8-connected foreground (non-zero) regions of `mask` and
    /// collect them into `components`.
    pub fn label(&mut self, mask: &[u8], w: usize, h: usize, components: &mut Vec<Component>) {
        debug_assert!(mask.len() >= w * h);
        components.clear();
        self.labels.clear();
        self.labels.resize(w * h, 0);
        self.parent.clear();
        self.parent.push(0); // dummy: provisional labels start at 1

        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let idx = row + x;
                if mask[idx] == 0 {
                    continue;
                }

                let mut label = 0u32;
                let mut merge = |parent: &mut Vec<u32>, other: u32| {
                    if other == 0 {
                        return;
                    }
                    if label == 0 {
                        label = other;
                    } else if other != label {
                        union(parent, label, other);
                    }
                };

                if x > 0 {
                    merge(&mut self.parent, self.labels[idx - 1]);
                }
                if y > 0 {
                    let up = idx - w;
                    if x > 0 {
                        merge(&mut self.parent, self.labels[up - 1]);
                    }
                    merge(&mut self.parent, self.labels[up]);
                    if x + 1 < w {
                        merge(&mut self.parent, self.labels[up + 1]);
                    }
                }

                if label == 0 {
                    label = self.parent.len() as u32;
                    self.parent.push(label);
                }
                self.labels[idx] = label;
            }
        }

        self.comp_of_root.clear();
        self.comp_of_root.resize(self.parent.len(), u32::MAX);
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let idx = row + x;
                let l = self.labels[idx];
                if l == 0 {
                    continue;
                }
                let root = find(&mut self.parent, l);
                let ci = if self.comp_of_root[root as usize] == u32::MAX {
                    let ci = components.len() as u32;
                    self.comp_of_root[root as usize] = ci;
                    components.push(Component {
                        area: 0,
                        min_x: x as u32,
                        min_y: y as u32,
                        max_x: x as u32,
                        max_y: y as u32,
                    });
                    ci
                } else {
                    self.comp_of_root[root as usize]
                };
                let c = &mut components[ci as usize];
                c.area += 1;
                c.min_x = c.min_x.min(x as u32);
                c.max_x = c.max_x.max(x as u32);
                // min_y is set at creation (row order); max_y only grows.
                c.max_y = y as u32;
                self.labels[idx] = ci + 1;
            }
        }
    }
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra < rb {
        parent[rb as usize] = ra;
    } else if rb < ra {
        parent[ra as usize] = rb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn label(mask: &[u8], w: usize, h: usize) -> Vec<Component> {
        let mut ccl = ConnectedComponents::new();
        let mut comps = Vec::new();
        ccl.label(mask, w, h, &mut comps);
        comps
    }

    #[test]
    fn empty_mask_has_no_components() {
        assert!(label(&[0u8; 100], 10, 10).is_empty());
    }

    #[test]
    fn single_blob_area_and_bbox() {
        let (w, h) = (20, 15);
        let mut mask = vec![0u8; w * h];
        for y in 3..9 {
            for x in 5..12 {
                mask[y * w + x] = 255;
            }
        }
        let comps = label(&mask, w, h);
        assert_eq!(comps.len(), 1);
        let c = comps[0];
        assert_eq!(c.area, 6 * 7);
        assert_eq!((c.min_x, c.min_y, c.max_x, c.max_y), (5, 3, 11, 8));
    }

    #[test]
    fn diagonal_pixels_are_one_component() {
        let (w, h) = (8, 8);
        let mut mask = vec![0u8; w * h];
        for i in 0..5 {
            mask[i * w + i] = 255;
        }
        let comps = label(&mask, w, h);
        assert_eq!(comps.len(), 1, "8-connectivity joins diagonals");
        assert_eq!(comps[0].area, 5);
    }

    #[test]
    fn separate_blobs_are_separate_components() {
        let (w, h) = (16, 8);
        let mut mask = vec![0u8; w * h];
        mask[w + 1] = 255; // (1,1)
        for y in 4..7 {
            for x in 10..14 {
                mask[y * w + x] = 255;
            }
        }
        let comps = label(&mask, w, h);
        assert_eq!(comps.len(), 2);
        assert_eq!(comps[0].area, 1);
        assert_eq!(comps[1].area, 12);
    }

    #[test]
    fn u_shape_merges_into_one_component() {
        let (w, h) = (10, 10);
        let mut mask = vec![0u8; w * h];
        for y in 0..8 {
            mask[y * w + 2] = 255;
            mask[y * w + 7] = 255;
        }
        for x in 2..8 {
            mask[8 * w + x] = 255;
        }
        let comps = label(&mask, w, h);
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].area, 8 + 8 + 6);
        let c = comps[0];
        assert_eq!((c.min_x, c.min_y, c.max_x, c.max_y), (2, 0, 7, 8));
    }

    #[test]
    fn labels_map_back_to_component_indices() {
        let (w, h) = (6, 4);
        let mut mask = vec![0u8; w * h];
        mask[0] = 255;
        mask[w * 2 + 4] = 255;
        let mut ccl = ConnectedComponents::new();
        let mut comps = Vec::new();
        ccl.label(&mask, w, h, &mut comps);
        assert_eq!(comps.len(), 2);
        assert_eq!(ccl.labels()[0], 1);
        assert_eq!(ccl.labels()[w * 2 + 4], 2);
        assert_eq!(ccl.labels()[1], 0);
    }
}

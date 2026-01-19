//! A geomtric primitive used to construct fundamental spatial relations, such as precedence relations, and mining width relations.

use crate::relation_provider::RelationProvider;
use rstar::{AABB, RTree, RTreeObject};
use std::collections::{HashMap, HashSet};

/// A circle in 3D space, defined by its center and radius.
///
/// The circle lies in the XY plane at the Z coordinate of the center.
#[derive(Copy, Clone, Debug)]
pub struct Circle {
    pub center: [f32; 3],
    pub radius: f32,
}

impl Circle {
    /// Create a new circle with the given center and radius.
    ///
    /// # Arguments
    /// * `center` - A 3D point representing the center of the circle.
    /// * `radius` - The radius of the circle.
    #[inline(always)]
    pub fn new(center: [f32; 3], radius: f32) -> Self {
        Self { center, radius }
    }

    /// Generate a set of circles that cover a given point with a given radius.
    ///
    /// # Arguments
    /// * `point` - A 3D point representing the center of the area to cover.
    /// * `radius` - The radius of the area to cover.
    /// * `resolution` - A 2D vector representing the spacing between the centers of the covering circles.
    #[inline(always)]
    pub fn covering(point: [f32; 3], radius: f32, resolution: [f32; 2]) -> Vec<Self> {
        let x_start = point[0] - radius;
        let y_start = point[1] - radius;

        let x_end = point[0] + radius;
        let y_end = point[1] + radius;

        let nx = ((x_end - x_start) / resolution[0]).ceil() as usize;
        let ny = ((y_end - y_start) / resolution[1]).ceil() as usize;

        let mut circles = Vec::new();
        for xi in 0..=nx {
            for yi in 0..=ny {
                let x = x_start + xi as f32 * resolution[0];
                let y = y_start + yi as f32 * resolution[1];

                if (x - point[0]).powi(2) + (y - point[1]).powi(2) <= radius.powi(2) {
                    circles.push(Self::new([x, y, point[2]], radius));
                }
            }
        }

        circles
    }

    /// Get the axis-aligned bounding box (AABB) of the circle.
    #[inline(always)]
    pub fn aabb(&self) -> AABB<[f32; 3]> {
        let lower = [
            self.center[0] - self.radius,
            self.center[1] - self.radius,
            self.center[2],
        ];
        let upper = [
            self.center[0] + self.radius,
            self.center[1] + self.radius,
            self.center[2],
        ];
        AABB::from_corners(lower, upper)
    }

    /// Get the vertically padded axis-aligned bounding box (AABB) of the circle.
    ///
    /// # Arguments
    /// * `padding` - The amount of vertical padding to add above and below the circle.
    #[inline(always)]
    fn v_padded_aabb(&self, padding: f32) -> AABB<[f32; 3]> {
        let lower = [
            self.center[0] - self.radius,
            self.center[1] - self.radius,
            self.center[2] - padding,
        ];
        let upper = [
            self.center[0] + self.radius,
            self.center[1] + self.radius,
            self.center[2] + padding,
        ];
        AABB::from_corners(lower, upper)
    }

    /// Check if the circle contains a given point.
    ///
    /// # Arguments
    /// * `point` - A 3D point to check for containment within the circle.
    #[inline(always)]
    pub fn contains(&self, point: [f32; 3]) -> bool {
        let dx = point[0] - self.center[0];
        let dy = point[1] - self.center[1];
        dx * dx + dy * dy <= self.radius * self.radius //&& dz == 0.0
    }

    /// Build predecessor relations based on the circle geometry.
    ///
    /// This will build a cone of relations upwards from each block, with the slope defined by the slope_angle.
    /// The number of benches defines how many levels of blocks to consider above each block.
    ///
    /// # Arguments
    /// * `block_inds` - A slice of 3D integer indices representing the blocks.
    /// * `block_size` - A 3D vector representing the size of each block
    /// * `slope_angle` - The angle of the slope in radians.
    /// * `n_benches` - The number of benches to consider above each block.
    pub fn build_pred_relations(
        block_inds: &[[i32; 3]],
        block_size: [f32; 3],
        slope_angle: f32,
        n_benches: u32,
    ) -> RelationProvider {
        #[derive(Clone)]
        pub struct IndexedBlock {
            ind: usize,
            location: [f32; 3],
        }

        impl RTreeObject for IndexedBlock {
            type Envelope = AABB<[f32; 3]>;

            fn envelope(&self) -> Self::Envelope {
                AABB::from_point(self.location)
            }
        }

        let indexed_blocks = block_inds
            .iter()
            .copied()
            .enumerate()
            .map(|(ind, [i, j, k])| {
                let location = [
                    i as f32 * block_size[0],
                    j as f32 * block_size[1],
                    k as f32 * block_size[2],
                ];

                IndexedBlock { ind, location }
            })
            .collect::<Vec<_>>();

        let tree = RTree::bulk_load(indexed_blocks.clone());

        let mut preds = RelationProvider::default();
        for block in indexed_blocks.iter() {
            let radius_step = block_size[2] * f32::tan(slope_angle);
            let mut tmp = vec![];
            for bench in 1..=n_benches {
                let mut envelope = Self::new(block.location, radius_step * bench as f32);
                envelope.center[2] += block_size[2] * bench as f32;
                let aabb = envelope.v_padded_aabb(block_size[2] / 2.0);

                for candidate in tree.locate_in_envelope_intersecting(&aabb) {
                    if envelope.contains(candidate.location) {
                        tmp.push(candidate.ind as u32)
                    }
                }
            }
            preds.add_relations(tmp);
        }

        preds
    }

    /// Build mining width relations based on the circle geometry.
    ///
    /// This build a graph where each block is connected to all other blocks within a given radius on the same level.
    ///
    /// # Arguments
    /// * `block_inds` - A slice of 3D integer indices representing the blocks.
    /// * `block_size` - A 3D vector representing the size of each block
    /// * `radius` - The radius within which to connect blocks.
    pub fn build_mw_relations(
        block_inds: &[[i32; 3]],
        block_size: [f32; 3],
        radius: f32,
    ) -> RelationProvider<(u32, f32)> {
        #[derive(Clone)]
        pub struct IndexedBlock {
            ind: usize,
            location: [f32; 3],
        }

        impl RTreeObject for IndexedBlock {
            type Envelope = AABB<[f32; 3]>;

            fn envelope(&self) -> Self::Envelope {
                AABB::from_point(self.location)
            }
        }

        let indexed_blocks = block_inds
            .iter()
            .copied()
            .enumerate()
            .map(|(ind, [i, j, k])| {
                let location = [
                    i as f32 * block_size[0],
                    j as f32 * block_size[1],
                    k as f32 * block_size[2],
                ];

                IndexedBlock { ind, location }
            })
            .collect::<Vec<_>>();

        let tree = RTree::bulk_load(indexed_blocks.clone());

        let mut mw = RelationProvider::default();
        let mut cnts = HashSet::new();
        for block in indexed_blocks.iter() {
            let mut tmp = vec![];

            let envelope = Self::new(block.location, radius);

            let aabb = envelope.v_padded_aabb(block_size[2] / 2.0);

            for candidate in tree.locate_in_envelope_intersecting(&aabb) {
                if candidate.location[2] != block.location[2] {
                    panic!("Levels do not match in MW relation building!")
                }
                if envelope.contains(candidate.location)
                    && candidate.location[2] == block.location[2]
                {
                    let dist = f32::sqrt(
                        (candidate.location[0] - envelope.center[0]).powi(2)
                            + (candidate.location[1] - envelope.center[1]).powi(2),
                    );
                    tmp.push((candidate.ind as u32, dist))
                }
            }

            if cnts.insert(tmp.len()) {
                println!("MW relations count: {}", tmp.len());
            }
            mw.add_relations(tmp);
        }
        mw
    }

    /// Get an iterator over the integer indices contained within the circle.
    pub fn indices(&self) -> CircleIterator {
        CircleIterator::new(*self)
    }
}

/// An iterator over the integer indices contained within a circle.
pub struct CircleIterator {
    center: [f32; 2],
    radius: f32,
    current: [i32; 2],
    bounds: (i32, i32, i32, i32), // min_x, max_x, min_y, max_y
}

impl CircleIterator {
    /// Create a new iterator for the given circle.
    pub fn new(circle: Circle) -> Self {
        let min_x = (circle.center[0] - circle.radius).floor() as i32;
        let max_x = (circle.center[0] + circle.radius).ceil() as i32;
        let min_y = (circle.center[1] - circle.radius).floor() as i32;
        let max_y = (circle.center[1] + circle.radius).ceil() as i32;

        Self {
            center: [circle.center[0], circle.center[1]],
            radius: circle.radius,
            current: [min_x, min_y],
            bounds: (min_x, max_x, min_y, max_y),
        }
    }
}

impl Iterator for CircleIterator {
    type Item = [i32; 2];

    fn next(&mut self) -> Option<Self::Item> {
        while self.current[1] <= self.bounds.3 {
            let x = self.current[0];
            let y = self.current[1];

            self.current[0] += 1;
            if self.current[0] > self.bounds.1 {
                self.current[0] = self.bounds.0;
                self.current[1] += 1;
            }

            let dx = x as f32 - self.center[0];
            let dy = y as f32 - self.center[1];
            if dx * dx + dy * dy <= self.radius * self.radius {
                return Some([x, y]);
            }
        }
        None
    }
}

/// An ellipsoid in 2D space, defined by its center and semi-axes.
pub struct Ellipsoid {
    pub center: (i32, i32),
    pub semi_axes: (i32, i32),
}

impl Ellipsoid {
    /// Create a new ellipsoid with the given center and semi-axes.
    pub fn new(center: (i32, i32), semi_axes: (i32, i32)) -> Self {
        Self { center, semi_axes }
    }

    /// Check if the ellipsoid contains a given point.
    #[inline(always)]
    pub fn contains_point(&self, (x, y): (i32, i32)) -> bool {
        let (cx, cy) = self.center;
        let (a, b) = self.semi_axes;
        let dx = x - cx;
        let dy = y - cy;

        // Equation of ellipse: (dx/a)^2 + (dy/b)^2 <= 1
        (dx * dx) as f64 / (a * a) as f64 + (dy * dy) as f64 / (b * b) as f64 <= 1.0
    }

    /// Build predecessor and successor relations based on the ellipsoid geometry.
    ///
    /// This will build relations between blocks based on their positions in 3D space,
    pub fn build_pred_succ_relations(
        bench_step: i32,
        blocks: &[[i32; 3]],
    ) -> (RelationProvider, RelationProvider) {
        fn gen_iter(
            i: i32,
            j: i32,
            k: i32,
            v_offset: i32,
            bench_step: i32,
            map: &HashMap<[i32; 3], u32>,
        ) -> impl Iterator<Item = [i32; 3]> + '_ {
            (1..=v_offset.abs()).flat_map(move |delta| {
                EllipsoidIndices::new((i, j), (bench_step * delta, bench_step * delta))
                    .map(move |(i, j)| [i, j, k + delta * v_offset.signum()])
                    .filter(|ind| map.contains_key(ind))
                    .collect::<Vec<_>>()
            })
            // .take_while(|inds| !inds.is_empty())
            // .flatten()
        }
        // 1. Create a [`HashMap`] of the blocks, using the locations as keys.
        let map = blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (*b, i as u32))
            .collect::<HashMap<_, _>>();

        assert!(map.len() == blocks.len());

        // 2. Build pred relations.
        let mut preds = RelationProvider::default();
        let mut succs = RelationProvider::default();
        for block in blocks.iter() {
            let inds = gen_iter(block[0], block[1], block[2], 5, bench_step, &map)
                .filter_map(|key| map.get(&key).copied())
                .collect::<Vec<_>>();

            preds.add_relations(inds);

            let inds = gen_iter(block[0], block[1], block[2], -5, bench_step, &map)
                .filter_map(|key| map.get(&key).copied())
                .collect::<Vec<_>>();

            succs.add_relations(inds);
        }

        (preds, succs)
    }

    /// Get an iterator over the integer indices contained within the ellipsoid.
    #[inline(always)]
    pub fn indices(&self) -> EllipsoidIndices {
        EllipsoidIndices::new(self.center, self.semi_axes)
    }
}

/// An iterator over the integer indices contained within an ellipsoid.
pub struct EllipsoidIndices {
    center: (i32, i32),
    semi_axes: (i32, i32),
    current: (i32, i32),
    bounds: ((i32, i32), (i32, i32)),
}

impl EllipsoidIndices {
    /// Create a new iterator for the given ellipsoid.
    #[inline(always)]
    pub fn new(center: (i32, i32), semi_axes: (i32, i32)) -> Self {
        let ((cx, cy), (a, b)) = (center, semi_axes);

        // Calculate the bounding box for the ellipsoid
        let bounds = ((cx - a, cx + a), (cy - b, cy + b));

        EllipsoidIndices {
            center,
            semi_axes,
            current: (bounds.0.0, bounds.1.0), // Start at the top-left corner of the bounding box
            bounds,
        }
    }

    /// Check if the given point is inside the ellipsoid.
    #[inline(always)]
    fn is_inside_ellipse(&self, x: i32, y: i32) -> bool {
        let (cx, cy) = self.center;
        let (a, b) = self.semi_axes;
        let dx = x - cx;
        let dy = y - cy;

        // Equation of ellipse: (dx/a)^2 + (dy/b)^2 <= 1
        (dx * dx) as f64 / (a * a) as f64 + (dy * dy) as f64 / (b * b) as f64 <= 1.0
    }
}

impl Iterator for EllipsoidIndices {
    type Item = (i32, i32);

    fn next(&mut self) -> Option<Self::Item> {
        let ((min_x, max_x), (_min_y, max_y)) = self.bounds;

        while self.current.1 <= max_y {
            let (x, y) = self.current;
            self.current.0 += 1; // Move to the next x

            if self.current.0 > max_x {
                // Move to the next row
                self.current.0 = min_x;
                self.current.1 += 1;
            }

            // println!("{:?} check result {}", (x, y), self.is_inside_ellipse(x, y));
            if self.is_inside_ellipse(x, y) {
                return Some((x, y));
            }
        }

        None // No more points inside the ellipse
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use itertools::Itertools;

    use super::*;

    #[test]
    fn zero_axis() {
        let el = Circle::new([1000.0, 100000.0, 10000.0], 0.75);
        for ind in el.indices() {
            println!("{:?}", ind);
        }
    }

    #[test]
    fn contains_self() {
        let val = 1000000.0;
        let el = Circle::new([val; 3], 0.0);
        assert!(el.contains([val; 3]));
    }

    #[test]
    fn matching_impls() {
        let mut circ = Circle::new([0.0; 3], 0.0);

        for _i in 0..40 {
            for ind in circ.indices() {
                assert!(circ.contains([ind[0] as f32, ind[1] as f32, 0.0]))
            }

            circ.radius += 0.75;
        }
    }

    #[test]
    fn contains() {
        let test = (0..200).cartesian_product(0..200);
        let ellipsoide = Ellipsoid::new((46, 131), (9, 9));

        println!("contains: {}", ellipsoide.contains_point((50, 139)));

        let mut s = HashSet::new();
        for ind in EllipsoidIndices::new((46, 131), (9, 9)) {
            if !ellipsoide.contains_point(ind) {
                panic!();
            }

            s.insert(ind);
        }

        for ind in test {
            if ellipsoide.contains_point(ind) && !s.contains(&ind) {
                panic!();
            }

            if !ellipsoide.contains_point(ind) && s.contains(&ind) {
                panic!()
            }
        }
    }

    #[test]
    fn iter_test() {
        println!("[");
        for (i, j) in EllipsoidIndices::new((10, 10), (10, 10)) {
            println!("[{i}, {j}],");
        }
        println!("]");
    }
    #[test]
    fn exclusive_inds() {
        let e1 = EllipsoidIndices::new((10, 10), (100, 100)).collect::<Vec<_>>();
        let e2 = EllipsoidIndices::new((11, 11), (100, 100)).collect::<Vec<_>>();

        let e1_exclusize = e1
            .iter()
            .filter(|e| !e2.iter().contains(e))
            .copied()
            .collect::<Vec<_>>();
        let e2_exclusize = e2
            .iter()
            .filter(|e| !e1.iter().contains(e))
            .copied()
            .collect::<Vec<_>>();

        println!("e1_len: {}", e1_exclusize.len());
        // println!("e1_len: {}, e1: {:?}", e1_exclusize.len(), e1_exclusize);
        println!("e2_len: {}", e2_exclusize.len());
        // println!("e2_len: {}, e2: {:?}", e2_exclusize.len(), e2_exclusize);
    }
}

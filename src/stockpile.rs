#[derive(Copy, Clone)]
pub struct StockPile {
    pub total_mass: f64,
    pub total_metal: f64,
}

#[derive(Copy, Clone)]
pub struct StockPileRehandle {
    pub rehandled_mass: f64,
    pub rehandled_metal: f64,
}

impl StockPile {
    pub fn empty() -> Self {
        Self {
            total_mass: 0.0,
            total_metal: 0.0,
        }
    }

    pub fn step(&mut self, input_mass: f64, input_metal: f64, rehandle: f64) -> StockPileRehandle {
        // 1. add inputs.
        self.total_mass += input_mass;
        self.total_metal += input_metal;

        // 2. Compute rehandle
        let rehandled_mass = self.total_mass * rehandle;
        let rehandled_metal = self.total_metal * rehandle;

        // 3. Subtract rehandled material.
        self.total_mass -= rehandled_mass;
        self.total_metal -= rehandled_metal;

        // 4. Return rehandle.
        StockPileRehandle {
            rehandled_mass,
            rehandled_metal,
        }
    }

    pub fn reset(&mut self) {
        self.total_mass = 0.0;
        self.total_metal = 0.0;
    }
}

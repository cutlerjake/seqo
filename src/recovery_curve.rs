//! Recovery curve for grade to recovery mapping.

/// A recovery curve maps grades to recovery values using linear interpolation.
#[derive(Clone)]
pub struct RecoveryCurve {
    grades: Vec<f64>,
    recovery: Vec<f64>,
}

impl RecoveryCurve {
    /// Creates a new `RecoveryCurve` with the given grades and recovery values.
    pub fn new(grades: Vec<f64>, recovery: Vec<f64>) -> Self {
        Self { grades, recovery }
    }

    /// Returns the recovery value for a given grade using linear interpolation.
    pub fn recovery(&self, grade: f64) -> f64 {
        if !grade.is_normal() {
            return 0.0;
        }
        let n = self.grades.len();

        // Handle bounds
        if grade <= self.grades[0] {
            return self.recovery[0];
        }
        if grade >= self.grades[n - 1] {
            return self.recovery[n - 1];
        }

        // Find the interval containing the grade
        for i in 0..n - 1 {
            if self.grades[i] <= grade && grade <= self.grades[i + 1] {
                let x0 = self.grades[i];
                let x1 = self.grades[i + 1];
                let y0 = self.recovery[i];
                let y1 = self.recovery[i + 1];

                // Linear interpolation formula
                return y0 + (y1 - y0) * (grade - x0) / (x1 - x0);
            }
        }

        println!("grade: {}", grade);
        unreachable!("Grade value not in expected range");
    }
}

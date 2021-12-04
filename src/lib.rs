use hash_histogram::HashHistogram;
use std::{fmt, io};
use std::fmt::{Display, Formatter};
use std::collections::HashSet;
use std::hash::Hash;
use std::io::Write;

pub struct ConfusionMatrix<L: Hash+Clone+Eq+Ord> {
    label_2_right: HashHistogram<L>,
    label_2_wrong: HashHistogram<L>,
}

impl <L: Hash+Clone+Eq+Ord> ConfusionMatrix<L> {
    pub fn new() -> ConfusionMatrix<L> {
        ConfusionMatrix {
            label_2_right: HashHistogram::new(),
            label_2_wrong: HashHistogram::new(),
        }
    }

    pub fn record(&mut self, img_label: L, classification: L) {
        if classification == img_label {
            self.label_2_right.bump(&img_label);
        } else {
            self.label_2_wrong.bump(&img_label);
        }
    }

    pub fn all_labels(&self) -> HashSet<L> {
        self.label_2_wrong.all_labels()
            .union(&self.label_2_right.all_labels())
            .cloned()
            .collect()
    }

    pub fn error_rate(&self) -> f64 {
        let total_right = self.label_2_right.total_count() as f64;
        let total_wrong = self.label_2_wrong.total_count() as f64;
        total_wrong / (total_right + total_wrong)
    }
}

impl <L: Display+Hash+Clone+Eq+Ord> fmt::Display for ConfusionMatrix<L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut ordered_labels: Vec<L> = self.all_labels().iter().cloned().collect();
        ordered_labels.sort_unstable();
        for label in ordered_labels {
            writeln!(f, "{}: {} correct, {} incorrect", label,
                     self.label_2_right.count(&label),
                     self.label_2_wrong.count(&label))?;
        }
        Ok(())
    }
}

pub trait Classifier<I,L:Hash+Clone+Eq+Ord> {
    fn train(&mut self, training_images: &Vec<(L,I)>);

    fn classify(&self, example: &I) -> L;

    fn test(&self, testing_images: &[(L,I)]) -> ConfusionMatrix<L> {
        let mut result = ConfusionMatrix::new();
        let mut count = 0;
        let twentieth = testing_images.len() / 20;
        for test_img in testing_images {
            result.record(test_img.0.clone(), self.classify(&test_img.1));
            count += 1;
            if count % twentieth == 0 {
                print!("{}%; ", count * 5 / twentieth);
                io::stdout().flush().expect("Could not flush stdout");
            }
        }
        println!();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix() {
        let mut matrix = ConfusionMatrix::new();

        let one_ok = 6;
        let one_er = 4;
        let two_ok = 7;
        let two_er = 3;

        for _ in 0..one_ok {
            matrix.record(1, 1);
        }

        for _ in 0..one_er {
            matrix.record(1, 2);
        }

        for _ in 0..two_ok {
            matrix.record(2, 2);
        }

        for _ in 0..two_er {
            matrix.record(2, 1);
        }

        assert_eq!(format!("1: {} correct, {} incorrect\n2: {} correct, {} incorrect\n", one_ok, one_er, two_ok, two_er), matrix.to_string());
    }
}
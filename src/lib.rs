#![doc = include_str!("../README.md")]

pub mod aggregate_perturbation;
pub mod block_perturbation;
pub mod circle;
pub mod geometric_constraint;
pub mod helpers;
mod idx_period_map;
pub mod local_search;
pub mod mine;
pub mod objective_function;
pub mod perturbation_summary;
pub mod recovery_curve;
pub mod relation_provider;
pub mod state_summary;
pub mod stockpile;
pub mod walkers;

pub use rayon;
pub use seqo_derive;
pub use simd;

pub mod prelude {
    pub use crate::aggregate_perturbation::*;
    pub use crate::block_perturbation::*;
    pub use crate::circle::*;
    pub use crate::geometric_constraint::*;
    pub use crate::helpers::*;
    pub use crate::local_search::*;
    pub use crate::mine::*;
    pub use crate::objective_function::*;
    pub use crate::perturbation_summary::*;
    pub use crate::recovery_curve::*;
    pub use crate::relation_provider::*;
    pub use crate::state_summary::*;
    pub use crate::stockpile::*;
    pub use crate::walkers::*;
}

pub fn init_opt_logger() {
    use env_logger::WriteStyle;
    use env_logger::fmt::style::{AnsiColor, Color};
    use log::{Level, LevelFilter};
    use std::io::Write;
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .write_style(WriteStyle::Auto)
        .format(|buf, record| {
            // ----- Level style -----
            let mut level_style = buf.default_level_style(record.level());
            match record.level() {
                Level::Error => level_style = level_style
                    .fg_color(Some(Color::Ansi(AnsiColor::Red)))
                    .bold(),
                Level::Warn => level_style = level_style
                    .fg_color(Some(Color::Ansi(AnsiColor::Yellow)))
                    .bold(),
                Level::Info => level_style = level_style.fg_color(Some(Color::Ansi(AnsiColor::Green))),
                Level::Debug => level_style = level_style.fg_color(Some(Color::Ansi(AnsiColor::Blue))),
                Level::Trace => level_style = level_style.fg_color(Some(Color::Ansi(AnsiColor::Magenta))),
            };

            // ----- Target (module path) -----
            let mut time_stye = buf.default_level_style(record.level());
            time_stye = time_stye.fg_color(Some(Color::Ansi(AnsiColor::Cyan)));
            let time = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.6f");

            // ----- File:line (dim gray) -----
            let mut site_style = buf.default_level_style(record.level());
            site_style = site_style.fg_color(Some(Color::Ansi(AnsiColor::Black)))
                .italic();
            // ----- Message (args) -----
            let mut msg_style = buf.default_level_style(record.level());
            msg_style = msg_style.fg_color(Some(Color::Ansi(AnsiColor::White)));

            writeln!(buf, "[{level_style}{level}{level_style:#}] {time_stye}{time}{time_stye:#} ({site_style}{file}:{line}{site_style:#}) {msg_style}{msg}{msg_style:#}",
                level = record.level(),
                file = record.file().unwrap_or("?"),
                line = record.line().unwrap_or(0),
                msg = record.args(),
            )
        })
        .init();
}

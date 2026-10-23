//! Print the current map whenever it changes.
//!
//!     cargo run --release -p mapleocr-core --example map_name

use mapleocr_core::{MapleOcr, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ocr = MapleOcr::auto()?; // finds the game window, calibrates on the first frame
    ocr.subscribe(&["map_name"], |r| {
        if let Value::Map(m) = &r.value {
            println!("{} / {}  (confidence {:.2}, {:.1} ms)", m.region, m.name, r.confidence, r.latency_ms());
        }
    })?;
    std::thread::park(); // readings arrive on mapleocr's worker thread
    Ok(())
}

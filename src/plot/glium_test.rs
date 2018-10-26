extern crate galerkin;
extern crate rand;

use galerkin::plot::glium::*;
use galerkin::distmesh::distmesh_2d::*;
use std::sync::mpsc::{self, Sender, Receiver};
use std::thread;

fn main() {
    let mesh = ellipse();
    let mesh_clone = ellipse();
    let f = move |sender: Sender<Vec<f64>>| {
        loop {
            let values = mesh_clone.points.iter()
                .map(|point| rand::random::<f64>())
                .collect();
            sender.send(values).unwrap();
            thread::sleep_ms(300);
        }
    };
    run_inside_plot(mesh, f);
}
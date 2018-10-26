extern crate glium;
extern crate rand;

use std::thread;
use glium::{glutin, Surface};
use distmesh::mesh::{Mesh, Point2D};
use std::sync::mpsc::{self, Sender, Receiver};
use std::f32;
use glium::uniforms::{Uniforms, UniformValue};

pub struct OpenGLPlot {
    mesh: Mesh,
    display: glium::Display,
    program: glium::Program,
    indices: glium::IndexBuffer<u32>,
    vertices: glium::VertexBuffer<Vertex>,
    matrix: [[f32; 4]; 4],
}

impl OpenGLPlot {
    pub fn new(mesh: Mesh, display: glium::Display, program: glium::Program) -> OpenGLPlot {
        let vertices: Vec<Vertex> = mesh.points.iter()
            .map(|&p| Vertex::from_point(p))
            .collect();
        let matrix = viewport_transform_matrix(&mesh);
        let vertex_buffer = glium::VertexBuffer::new(&display, &vertices).unwrap();
        let indices: Vec<u32> = mesh.triangles.iter()
            .flat_map(|tri| vec![tri.a as u32, tri.b as u32, tri.c as u32].into_iter())
            .collect();
        let index_buffer = glium::index::IndexBuffer::new(&display,
                                                          glium::index::PrimitiveType::TrianglesList,
                                                          &indices)
            .expect("could not create index buffer");
        OpenGLPlot { mesh, display, program, indices: index_buffer, vertices: vertex_buffer, matrix }
    }

    pub fn plot(&self, vertex_values: Vec<f64>) {
        let mut target = self.display.draw();
        let value_buffer = glium::VertexBuffer::new(&self.display, &vertex_values.iter()
            .map(|&x| Unknown { value: [x as f32, 0.0_f32] })
            .collect::<Vec<Unknown>>()).unwrap();
        let uniforms = uniform! {
            matrix: self.matrix,
            colorTransform: color_transform_matrix(&vertex_values),
        };
        target.draw(
            (&self.vertices, &value_buffer),
            &self.indices,
            &self.program,
            &uniforms,
            &Default::default(),
        ).unwrap();
        target.finish().unwrap();
    }
}

fn viewport_transform_matrix(mesh: &Mesh) -> [[f32; 4]; 4] {
    let (xmin, xmax, ymin, ymax) = mesh.points.iter()
        .fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(xmin, xmax, ymin, ymax), point: &Point2D| {
                (
                    xmin.min(point.x as f32),
                    xmax.max(point.x as f32),
                    ymin.min(point.y as f32),
                    ymax.max(point.y as f32)
                )
            });
    // 2(xmax) / (xmax - xmin) - (xmax + xmin) / (xmax - xmin) = xmax - xmin / xmax - xmin = 1.
    let result = [
        [2. / (xmax - xmin), 0., 0., 0.],
        [0., 2. / (ymax - ymin), 0., 0.],
        [0., 0., 1., 0.],
        [-(xmax + xmin) / (xmax - xmin), -(ymax + ymin) / (ymax - ymin), 0., 1.]
    ];
    result
}

fn color_transform_matrix(values: &Vec<f64>) -> [[f32; 4]; 4] {
    let (min, max) = values.iter().fold(
        (f32::MAX, f32::MIN),
        |(min, max): (f32, f32), &val| {
            (
                min.min(val as f32),
                max.max(val as f32)
            )
        });
    [
        [1. / (max - min), 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [-min / (max - min), 0., 0., 1.],
    ]
}

pub fn run_inside_plot<F>(mesh: Mesh, f: F)
    where
        F: Fn(Sender<Vec<f64>>) -> () + Send + Sync + 'static,
{
    let mut events_loop = glutin::EventsLoop::new();
    let window = glutin::WindowBuilder::new();
    let context = glutin::ContextBuilder::new();
    let display = glium::Display::new(window, context, &events_loop).unwrap();

    let program = glium::Program::new(
        &display,
        glium::program::SourceCode {
            vertex_shader: VERTEX_SHADER_SRC,
            fragment_shader: FRAGMENT_SHADER_SRC,
            geometry_shader: None,
            tessellation_control_shader: None,
            tessellation_evaluation_shader: None,
        },
    ).unwrap();

    let plot = OpenGLPlot::new(mesh, display, program);

    let (sender, receiver) = mpsc::channel();

    thread::spawn(move || f(sender));

    let mut closed = false;
    while !closed {
        // block on receiving values to plot
        let values = receiver.recv().unwrap();

        // plot the values
        plot.plot(values);

        // maybe close the window
        events_loop.poll_events(|ev| {
            match ev {
                glutin::Event::WindowEvent { event, .. } => match event {
                    glutin::WindowEvent::CloseRequested => closed = true,
                    _ => (),
                },
                _ => (),
            }
        });
    }
}

#[derive(Debug, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

implement_vertex!(Vertex, position);

#[derive(Copy, Clone)]
struct Unknown {
    value: [f32; 2],
}

implement_vertex!(Unknown, value);

impl Vertex {
    fn from_point(point: Point2D) -> Vertex {
        Vertex { position: [point.x as f32, point.y as f32] }
    }
}

const VERTEX_SHADER_SRC: &str = r#"
    #version 140

    in vec2 value;
    in vec2 position;
    out vec2 vertColor;

    uniform mat4 matrix;

    void main() {
        gl_Position = matrix * vec4(position, 0.0, 1.0);
        vertColor = value;
    }
"#;

const FRAGMENT_SHADER_SRC: &str = r#"
    #version 140

    in vec2 vertColor;
    out vec4 color;

    uniform mat4 colorTransform;

    void main() {
        color = colorTransform * vec4(vertColor, 0.0, 1.0);
    }
"#;

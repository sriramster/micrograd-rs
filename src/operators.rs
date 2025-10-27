use std::cell::RefCell;
use std::rc::Rc;

pub mod operators {
    use super::*;
    use std::fmt;
    use std::collections::HashSet;
    use std::ops::{Add, Mul, Div, Sub};
    
    #[derive(Clone)]
    pub struct GraphNode {
        pub data: f64,
        pub grad: f64,
        pub label: String,
        pub prev: Vec<Rc<RefCell<GraphNode>>>,
        pub op: Option<String>,
        pub backward: Option<Rc<dyn Fn()>>,
    }

    #[derive(Debug, Clone)]
    pub struct Value(Rc<RefCell<GraphNode>>);

    impl GraphNode {
        fn fmt_indented(&self, f: &mut fmt::Formatter<'_>, indent: usize) -> fmt::Result {
            let indent_str = " ".repeat(indent);
            writeln!(
                f,
                "{}{} (data={:.6}, grad={:.6}, op={:?})",
                indent_str,
                if self.label.is_empty() { "GraphNode" } else { &self.label },
                self.data,
                self.grad,
                self.op
            )?;

            for parent_rc in &self.prev {
                parent_rc.borrow().fmt_indented(f, indent + 4)?;
            }

            Ok(())
        }

        fn topological_sort(root : &Value) -> Vec<Value> {
            let mut topo: Vec<Value> = Vec::new();
            let mut visited: HashSet<usize> = HashSet::new();

            fn dfs(node_rc: Rc<RefCell<GraphNode>>, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
                let id = Rc::as_ptr(&node_rc) as usize;
                if visited.contains(&id) { return; }
                visited.insert(id);

                let parents: Vec<Rc<RefCell<GraphNode>>> = node_rc.borrow().prev.clone();

                for w in parents {
                    dfs(w, visited, topo);
                }

                topo.push(Value(node_rc.clone()));
            }

            dfs(root.rc(), &mut visited, &mut topo);
            topo
        }

        pub fn backward(root: &Value)  {
            let topo = GraphNode::topological_sort(root);
            root.borrow_mut().grad = 1.0;
            
            for node in topo.into_iter().rev() {
                if let Some(cb) = node.borrow().backward.as_ref() {
                    (cb)();
                }
            }
        }
    }

    impl fmt::Debug for GraphNode {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            writeln!(f, "Computation Graph:")?;
            self.fmt_indented(f, 0)
        }
    }

    impl Value {
        fn rc(&self) -> Rc<RefCell<GraphNode>> { self.0.clone() }

        pub fn new(data: f64, label: &str) -> Self {
            Value(Rc::new(RefCell::new(GraphNode {
                data,
                grad: 0.0,
                label: label.to_string(),
                prev: vec![],
                op: None,
                backward: None,
            })))
        }

        // Reference borrowing of the inner object
        pub fn borrow(&self) -> std::cell::Ref<GraphNode> {
            self.0.borrow()
        }

        // Create a mutable borrow of the inner object
        pub fn borrow_mut(&self) -> std::cell::RefMut<GraphNode> {
            self.0.borrow_mut()
        }

        pub fn label(&mut self, label: &str) {
            self.borrow_mut().label = label.to_string();
        }

        pub fn tanh(self) -> Value {
            let x = self.borrow().data;

            let out = Self::new(x.tanh(), "tanh");
            {
                let mut out_mut = out.borrow_mut();
                out_mut.op = Some("tanh".to_string());
                out_mut.prev = vec![Rc::clone(&self.0), ];
            }

            let weak_out = Rc::downgrade(&out.0);
            let weak_a = Rc::downgrade(&self.0);

            out.borrow_mut().backward = Some(Rc::new(move || {
                if let Some(out_rc) = weak_out.upgrade() {
                    let out_grad = out_rc.borrow().grad;
                    let out_val = out_rc.borrow().data;

                    if let Some(a_rc) = weak_a.upgrade() {
                        a_rc.borrow_mut().grad += (1.0 - out_val.powf(2.0)) * out_grad;
                    }
                }
            }));
            out
        }

        pub fn powop<T: Into<f64>>(self, other: T) -> Value {
            let exponent = other.into();
            let val = self.borrow().data.powf(exponent);
            let out = Self::new(val, "pow");
            {
                let mut out_mut = out.borrow_mut();
                out_mut.op = Some("pow".to_string());
                out_mut.prev = vec![Rc::clone(&self.0), ];
            }

            // Prepare references for gradient calculation
            let weak_out = Rc::downgrade(&out.0);
            let weak_a = Rc::downgrade(&self.0);

            out.borrow_mut().backward = Some(Rc::new(move || {
                if let Some(out_rc) = weak_out.upgrade() {
                    let out_grad = out_rc.borrow().grad;

                    // read current values of parents (they should exist)
                    if let Some(a_rc) = weak_a.upgrade() {
                        let a_val = a_rc.borrow().data;
                        a_rc.borrow_mut().grad += exponent * (a_val.powf((exponent - 1.0))) * out_grad;
                    }
                }
            }));
            out
        }
        
        pub fn exp(self) -> Value {
            let x = self.borrow().data;
            let out = Self::new(x.exp(), "exp");
            {
                let mut out_mut = out.borrow_mut();
                out_mut.op = Some("exp".to_string());
                out_mut.prev = vec![Rc::clone(&self.0), ];
            }

            let weak_out = Rc::downgrade(&out.0);
            let weak_a = Rc::downgrade(&self.0);

            out.borrow_mut().backward = Some(Rc::new(move || {
                if let Some(out_rc) = weak_out.upgrade() {
                    let out_grad = out_rc.borrow().grad;
                    let out_val = out_rc.borrow().data;

                    if let Some(a_rc) = weak_a.upgrade() {
                        a_rc.borrow_mut().grad += out_val * out_grad;
                    }
                }
            }));
            out
        }
    }

    impl From<f64> for Value {
        fn from(x: f64) -> Self {
            Value::new(x, "")
        }
    }

    impl Add for Value {
        type Output = Value;

        fn add (self, other: Value) -> Value {
            let sum = self.borrow().data + other.borrow().data;
            let out = Self::new(sum, "+");
            {
                let mut out_mut = out.borrow_mut();
                out_mut.op = Some("+".to_string());
                out_mut.prev = vec![Rc::clone(&self.0), Rc::clone(&other.0)];
            }

            // Capture weak refs for closure
            let weak_out = Rc::downgrade(&out.0);
            let weak_a = Rc::downgrade(&self.0);
            let weak_b = Rc::downgrade(&other.0);

            out.borrow_mut().backward = Some(Rc::new(move || {
                if let Some(out_rc) = weak_out.upgrade() {
                    let out_grad = out_rc.borrow().grad;
                    if let Some(a_rc) = weak_a.upgrade() {
                        a_rc.borrow_mut().grad += out_grad;
                    }

                    if let Some(b_rc) = weak_b.upgrade() {
                        b_rc.borrow_mut().grad += out_grad;
                    }
                }
            }));
            out
        }
    }

    impl Add<f64> for Value {
        type Output = Value;

        fn add(self, rhs: f64) -> Value {
            self + Value::from(rhs)
        }
    }

    impl<'a> Add<f64> for &'a Value {
        type Output = Value;

        fn add(self, rhs: f64) -> Value {
            self.clone() + Value::from(rhs)
        }
    }

    impl Mul for Value {
        type Output = Value;

        fn mul(self, other: Value) -> Value {
            let prod = self.borrow().data * other.borrow().data;

            let out = Self::new(prod, "*");
            {
                let mut out_mut = out.borrow_mut();
                out_mut.op = Some("*".to_string());
                out_mut.prev = vec![Rc::clone(&self.0), Rc::clone(&other.0)];
            }

            // backward closure for multiplication: d(a*b)/da = b, d(a*b)/db = a
            let weak_out = Rc::downgrade(&out.0);
            let weak_a = Rc::downgrade(&self.0);
            let weak_b = Rc::downgrade(&other.0);

            out.borrow_mut().backward = Some(Rc::new(move || {
                if let Some(out_rc) = weak_out.upgrade() {
                    let out_grad = out_rc.borrow().grad;

                    // read current values of parents (they should exist)
                    if let (Some(a_rc), Some(b_rc)) = (weak_a.upgrade(), weak_b.upgrade()) {
                        let a_val = a_rc.borrow().data;
                        let b_val = b_rc.borrow().data;

                        // accumulate gradients using product rule
                        a_rc.borrow_mut().grad += b_val * out_grad;
                        b_rc.borrow_mut().grad += a_val * out_grad;
                    }
                }
            }));

            out
        }
    }

    impl Mul<f64> for Value {
        type Output = Value;

        fn mul(self, rhs: f64) -> Value {
            self * Value::from(rhs)
        }
    }

    impl<'a> Mul<f64> for &'a Value {
        type Output = Value;

        fn mul(self, rhs: f64) -> Value {
            self.clone() * Value::from(rhs)
        }
    }

    impl Div for Value {
        type Output = Value;

        fn div (self, other: Value) -> Value {
            if other.borrow().data == 0.0 {
                panic!("Divide by zero")
            }
            self.clone() * other.powop(-1)
        }
    }

    impl Div<f64> for Value {
        type Output = Value;

        fn div(self, rhs: f64) -> Value {
            self * Value::from(rhs).powop(-1)
        }
    }

    impl<'a> Div<f64> for &'a Value {
        type Output = Value;

        fn div(self, rhs: f64) -> Value {
            self.clone() * Value::from(rhs).powop(-1)
        }
    }

    impl Sub for Value {
        type Output = Value;

        fn sub(self, other: Value) -> Value {
            self.clone() + (other * -1.0)
        }
    }

    impl Sub<f64> for Value {
        type Output = Value;

        fn sub(self, rhs: f64) -> Value {
            self + (Value::from(rhs) * -1.0)
        }
    }

    impl<'a> Sub<f64> for &'a Value {
        type Output = Value;

        fn sub(self, rhs: f64) -> Value {
            self.clone() + (Value::from(rhs) * -1.0 )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::operators::*;
    
    #[test]
    fn chaining() {
        let a = Value::new(2.0, "a");
        let b = Value::new(3.0, "b");
        let c = a.clone() + b.clone(); // c = a+b
        let d = c.clone() + a.clone(); // d = c + a = (a+b) + a = 2a + b

        GraphNode::backward(&d);
        println!("Chaining {:#?}", d.borrow());
    }
    
    #[test]
    fn simple_graph() {
        // x1, x2
        let x1 = Value::new(2.0, "x1");
        let x2 = Value::new(0.0, "x2");
        // w1, w2
        let w1 = Value::new(-3.0, "w1");
        let w2 = Value::new(1.0, "w2");

        let mut x1w1 = x1.clone() * w1.clone();
        x1w1.label("x1w1");

        let mut x2w2 = x2.clone() * w2.clone();
        x2w2.label("x2w2");

        let mut x1w1x2w2 = x1w1.clone() + x2w2.clone();
        x1w1x2w2.label("x1w1 + x2w2");

        // bias
        let b = Value::new(6.8813735870195432, "b");
        let mut n = x1w1x2w2.clone() + b.clone();
        n.label("n");

        let mut o = n.tanh();

        // Backward propagate
        GraphNode::backward(&o);
        println!("{:#?}", o.borrow());
    }

    #[test]
    fn simple_graph_2() {
        // x1, x2
        let x1 = Value::new(2.0, "x1");
        let x2 = Value::new(0.0, "x2");
        // w1, w2
        let w1 = Value::new(-3.0, "w1");
        let w2 = Value::new(1.0, "w2");

        let mut x1w1 = x1.clone() * w1.clone();
        x1w1.label("x1w1");

        let mut x2w2 = x2.clone() * w2.clone();
        x2w2.label("x2w2");

        let mut x1w1x2w2 = x1w1.clone() + x2w2.clone();
        x1w1x2w2.label("x1w1 + x2w2");

        // bias
        let b = Value::new(6.8813735870195432, "b");
        let mut n = x1w1x2w2.clone() + b.clone();
        n.label("n");

        let mul = Value::new(2.0, "mul");
        let mut e = (mul * n).exp();
        e.label("exponent");
        let mut i = (e.clone() - Value::new(1.0, "")) / (e + Value::new(1.0, ""));
        i.label("output");
        
        // Backward propagate
        GraphNode::backward(&i);
        println!("{:#?}", i.borrow());
    }

    #[test]
    fn div() {
        let a = Value::new(2.0, "a");
        let b = Value::new(3.0, "b");
        let mut c = b / a;
        c.label("c");
        GraphNode::backward(&c);
        println!("Div Op {:#?}", c.borrow());
    }

    #[test]
    fn pow() {
        let a = Value::new(2.0, "a");
        let mut c = a.powop(2);
        c.label("c");
        GraphNode::backward(&c);
        println!("Pow Op {:#?}", c.borrow());
    }

    
    #[test]
    fn sub() {
        let a = Value::new(2.0, "a");
        let b = Value::new(3.0, "b");
        let mut c = a - b;
        c.label("c");
        GraphNode::backward(&c);
        println!("Div Op {:#?}", c.borrow());
    }
    
    #[test]
    fn fail() {
        let a = Value::new(3.0, "a");
        let mut b = a.clone() + a.clone();
        b.label("b");
        GraphNode::backward(&b);
        println!("{:#?}", b.borrow());
    }
    
    #[test]
    fn it_works() {
        let a = Value::new(2.0, "a");
        let b = Value::new(-3.0, "b");
        let c = Value::new(10.0, "c");
        let mut d = a.clone() * b.clone() + c.clone();
        d.label("d");
        GraphNode::backward(&d);
        println!("{:#?}", d.borrow());
    }

    #[test]
    fn scalar() {
        let a = Value::new(2.0, "a");
        let b = -3.0;
        let mut d = a.clone() + b;
        d.label("d");
        GraphNode::backward(&d);
        println!("{:#?}", d.borrow());
    }

}

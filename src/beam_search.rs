use std::cmp::Ordering;
use std::collections::{BTreeSet, VecDeque};
use std::rc::Rc;

pub struct Hypothesis {
    pub prev: Option<Rc<Self>>,
    pub data: u32,
    pub cost: f32,
}

impl Hypothesis {
    pub fn new(prev: &Option<Rc<Hypothesis>>, prob: f32, data: u32) -> Rc<Self> {
        Rc::new(Self {
            prev: prev.as_ref().map(|prev| Rc::clone(prev)),
            data: data,
            cost: if let Some(prev) = prev {
                prev.cost - prob.ln()
            } else {
                -prob.ln()
            },
        })
    }

    pub fn gen_vec(&self, bos: u32) -> Vec<u32> {
        let mut stack = VecDeque::new();
        stack.push_front(self.data);
        let mut hyp = self.prev.as_ref().map(|hyp| Rc::clone(hyp));
        while let Some(h) = hyp.as_ref() {
            stack.push_front(h.data);
            hyp = h.prev.as_ref().map(|hyp| Rc::clone(hyp));
        }
        stack.push_front(bos);
        stack.into_iter().collect()
    }
}

impl PartialEq for Hypothesis {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for Hypothesis {}

impl PartialOrd for Hypothesis {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.cost.partial_cmp(&other.cost)
    }
}

impl Ord for Hypothesis {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

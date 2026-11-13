//! 纯 CPU 极速微型神经网络内核 (Micro-Neural Network Core)。
//!
//! 参数量极度紧凑（几十个标量参数，小于 1KB 内存），完全放入 CPU L1 缓存与寄存器。
//! 纯 Rust 原生数组运算，零外部动态堆分配，单次前向推理仅需几纳秒。

use serde::{Deserialize, Serialize};

/// 微型神经拓扑网络：输入 5 维 (IDF, LenNorm, Span, Order, SemanticDir)
/// -> 隐藏层 8 维 (Gated Linear Unit) -> 输出 1 维 (Sigmoid)。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MicroNet {
    /// 隐藏层权重 (8x5)
    pub w1: Vec<f32>,
    /// 门控权重 (8x5)
    pub w_gate: Vec<f32>,
    /// 隐藏层偏置 (8)
    pub b1: Vec<f32>,
    /// 输出层权重 (1x8)
    pub w2: Vec<f32>,
    /// 输出层偏置
    pub b2: f32,
}

impl Default for MicroNet {
    fn default() -> Self {
        Self {
            w1: vec![0.5; 40],
            w_gate: vec![0.1; 40],
            b1: vec![0.0; 8],
            w2: vec![0.2; 8],
            b2: 0.0,
        }
    }
}

impl MicroNet {
    /// 纯 CPU 纳秒级极速前向推理：输入 5 维特征。
    #[inline(always)]
    pub fn forward(&self, x: &[f32; 5]) -> f32 {
        let mut hidden = [0.0f32; 8];

        for i in 0..8 {
            let offset = i * 5;
            let dot = self.w1[offset] * x[0]
                + self.w1[offset + 1] * x[1]
                + self.w1[offset + 2] * x[2]
                + self.w1[offset + 3] * x[3]
                + self.w1[offset + 4] * x[4];

            let gate_dot = self.w_gate[offset] * x[0]
                + self.w_gate[offset + 1] * x[1]
                + self.w_gate[offset + 2] * x[2]
                + self.w_gate[offset + 3] * x[3]
                + self.w_gate[offset + 4] * x[4];

            let gate = 1.0 / (1.0 + (-gate_dot).exp());
            hidden[i] = (dot + self.b1[i]) * gate;
        }

        let mut out = self.b2;
        for i in 0..8 {
            out += self.w2[i] * hidden[i];
        }
        1.0 / (1.0 + (-out).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_net_forward_works_with_5d_input() {
        let net = MicroNet::default();
        let input = [1.0, 0.5, 0.2, 1.0, 0.8];
        let prob = net.forward(&input);
        assert!(prob > 0.0 && prob < 1.0);
    }
}

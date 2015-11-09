//
//  Layer.swift
//  ConvNetSwift
//
import Foundation

struct ParamsAndGrads {
    var params: [Double]
    var grads: [Double]
    var l1_decay_mul: Double?
    var l2_decay_mul: Double?
    
    init (
        inout params: [Double],
        inout grads: [Double],
        l1_decay_mul: Double,
        l2_decay_mul: Double) {
            self.params = params
            self.grads = grads
            self.l1_decay_mul = l1_decay_mul
            self.l2_decay_mul = l2_decay_mul
    }
}

protocol Layer {
    //    var sx: Int {get set}
    //    var sy: Int {get set}
    //
    //    var in_sx: Int {get set}
    //    var in_sy: Int {get set}
    //    var in_depth: Int {get set}
    var out_sx: Int {get set}
    var out_sy: Int {get set}
    var out_depth: Int {get set}
    //
    //    var stride: Int {get set}
    //    var pad: Int {get set}
    //
    //    var l1_decay_mul: Double {get set}
    //    var l2_decay_mul: Double {get set}
    //
    var layer_type: LayerType {get set}
    //    var filters: [Vol] {get set}
    //
    //    var biases: Vol {get set}
    //    var in_act: Vol {get set}
    var out_act: Vol? {get set}
    //
    //    var num_inputs: Int {get set}
    
    func getParamsAndGrads() -> [ParamsAndGrads]
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads])
    func forward(inout V: Vol, is_training: Bool) -> Vol
    func toJSON() -> [String: AnyObject]
}

protocol InnerLayer: Layer {
    func backward()
}

enum LayerType: String {
    case input = "input"
    case svm = "svm"
    case fc = "fc"
    case regression = "regression"
    case conv = "conv"
    case dropout = "dropout"
    case pool = "pool"
    case relu = "relu"
    case sigmoid = "sigmoid"
    case tanh = "tanh"
    case maxout = "maxout"
    case softmax = "softmax"
    case lrn = "lrn"
}

enum ActivationType: String {
    case undefined = "undefined"
    case relu = "relu"
    case sigmoid = "sigmoid"
    case tanh = "tanh"
    case maxout = "maxout"
}
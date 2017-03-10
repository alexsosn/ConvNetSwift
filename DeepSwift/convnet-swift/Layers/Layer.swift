//
//  Layer.swift
//  ConvNetSwift
//
import Foundation

public struct ParamsAndGrads {
    var params: [Double]
    var grads: [Double]
    var l1DecayMul: Double?
    var l2DecayMul: Double?
    
    init(params: inout [Double],
         grads: inout [Double],
         l1DecayMul: Double,
         l2DecayMul: Double) {
            self.params = params
            self.grads = grads
            self.l1DecayMul = l1DecayMul
            self.l2DecayMul = l2DecayMul
    }
}

public protocol Layer {
    var outSx: Int {get set}
    var outSy: Int {get set}
    var outDepth: Int {get set}
    var layerType: LayerType {get set}
    var outAct: Vol? {get set}
    func getParamsAndGrads() -> [ParamsAndGrads]
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads])
    func forward(_ vol: inout Vol, isTraining: Bool) -> Vol
    func toJSON() -> [String: AnyObject]
}

public protocol InnerLayer: Layer {
    func backward()
}

public enum  LayerType: String {
    case Input = "input"
    case SVM = "svm"
    case FC = "fc"
    case Regression = "regression"
    case Conv = "conv"
    case Dropout = "dropout"
    case Pool = "pool"
    case ReLU = "relu"
    case Sigmoid = "sigmoid"
    case Tanh = "tanh"
    case Maxout = "maxout"
    case Softmax = "softmax"
    case LRN = "lrn"
}

public enum  ActivationType: String {
    case Undefined = "undefined"
    case ReLU = "relu"
    case Sigmoid = "sigmoid"
    case Tanh = "tanh"
    case Maxout = "maxout"
}

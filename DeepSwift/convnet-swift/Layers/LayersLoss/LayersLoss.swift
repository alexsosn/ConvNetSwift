
// Layers that implement a loss. Currently these are the layers that
// can initiate a backward() pass. In future we probably want a more
// flexible system that can accomodate multiple losses to do multi-task
// learning, and stuff like that. But for now, one of the layers in this
// file must be the final layer in a Net.

// This is a classifier, with N discrete classes from 0 to N-1
// it gets a stream of N incoming numbers and computes the softmax
// function (exponentiate and normalize to sum to 1 as probabilities should)
import Foundation

protocol LossLayer: Layer {
    func backward(_ y: Int) -> Double
}

struct SoftmaxLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .Softmax
    
    var inSx: Int = 0
    var inSy: Int = 0
    var inDepth: Int = 0
    var numClasses: Int
    
    init (numClasses: Int) {
        self.numClasses = numClasses
    }
}

class SoftmaxLayer: LossLayer {
    
    var numInputs: Int
    var outDepth: Int
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var inAct: Vol?
    var outAct: Vol?
    
    var es: [Double] = []
    
    init(opt: SoftmaxLayerOpt) {
        // computed
        numInputs = opt.inSx * opt.inSy * opt.inDepth
        outDepth = numInputs
        outSx = 1
        outSy = 1
        layerType = .Softmax
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        
        let A = Vol(sx: 1, sy: 1, depth: outDepth, c: 0.0)
        
        // compute max activation
        var a_s = V.w
        let amax = V.w.max()!
        
        // compute exponentials (carefully to not blow up)
        var es = ArrayUtils.zerosDouble(outDepth)
        var esum = 0.0
        for i in 0 ..< outDepth {
            
            let e = exp(a_s[i] - amax)
            esum += e
            es[i] = e
        }
        
        // normalize and output to sum to one
        for i in 0 ..< outDepth {
            
            es[i] /= esum
            A.w[i] = es[i]
        }
        
        self.es = es // save these for backprop
        outAct = A
        return outAct!
    }
    
    func backward(_ y: Int) -> Double {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        x.dw = ArrayUtils.zerosDouble(x.w.count) // zero out the gradient of input Vol
        
        for i in 0 ..< outDepth {
            
            let indicator = i == y ? 1.0 : 0.0
            let mul = -(indicator - es[i])
            x.dw[i] = mul
        }
        // loss is the class negative log likelihood
        return -log(es[y])
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        json["numInputs"] = numInputs as AnyObject?
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        outDepth = json["outDepth"]
    //        outSx = json["outSx"]
    //        outSy = json["outSy"]
    //        layerType = json["layerType"]
    //        numInputs = json["numInputs"]
    //    }
}

// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.

struct RegressionLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .Regression
    
    var inSx: Int
    var inSy: Int
    var inDepth: Int
    var numNeurons: Int
    
    init(numNeurons: Int) {
        // warning: creativity!
        inSx = 1
        inSy = 1
        inDepth = 1
        self.numNeurons = numNeurons
        
    }
}

class RegressionLayer: LossLayer {
    
    var numInputs: Int
    var outDepth: Int
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: RegressionLayerOpt) {
        
        // computed
        numInputs = opt.inSx * opt.inSy * opt.inDepth
        outDepth = numInputs
        outSx = 1
        outSy = 1
        layerType = .Regression
    }

    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        outAct = V
        return V // identity function
    }

    // y is a list here of size numInputs
    // or it can be a number if only one value is regressed
    // or it can be a struct {dim: i, val: x} where we only want to
    // regress on dimension i and asking it to have value x
    
    func backward(_ y: [Double]) -> Double {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        x.dw = ArrayUtils.zerosDouble(x.w.count) // zero out the gradient of input Vol
        var loss = 0.0
        for i in 0 ..< outDepth {
            let dy = x.w[i] - y[i]
            x.dw[i] = dy
            loss += 0.5*dy*dy
        }
        inAct = x
        return loss
    }
    
    func backward(_ y: Double) -> Double {
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        x.dw = ArrayUtils.zerosDouble(x.w.count) // zero out the gradient of input Vol
        var loss = 0.0
        // lets hope that only one number is being regressed
        let dy = x.w[0] - y
        x.dw[0] = dy
        loss += 0.5*dy*dy
        return loss
    }
    
    func backward(_ y: Int) -> Double {
        return backward(Double(y))
    }
    
    struct Pair {
        var dim: Int
        var val: Double
    }
    
    func backward(_ y: Pair) -> Double {
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        x.dw = ArrayUtils.zerosDouble(x.w.count) // zero out the gradient of input Vol
        var loss = 0.0
        // assume it is a struct with entries .dim and .val
        // and we pass gradient only along dimension dim to be equal to val
        let i = y.dim
        let yi = y.val
        let dy = x.w[i] - yi
        x.dw[i] = dy
        loss += 0.5*dy*dy
        return loss
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        json["numInputs"] = numInputs as AnyObject?
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        outDepth = json["outDepth"]
    //        outSx = json["outSx"]
    //        outSy = json["outSy"]
    //        layerType = json["layerType"]
    //        numInputs = json["numInputs"]
    //    }
}

struct SVMLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .SVM
    
    var inSx: Int
    var inSy: Int
    var inDepth: Int
    var numClasses: Int
}

class SVMLayer: LossLayer {
    
    var numInputs: Int
    var outDepth: Int
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: SVMLayerOpt){
        // computed
        numInputs = opt.inSx * opt.inSy * opt.inDepth
        outDepth = numInputs
        outSx = 1
        outSy = 1
        layerType = .SVM
    }
    
    func forward(_ V: inout Vol, isTraining: Bool) -> Vol {
        inAct = V
        outAct = V // nothing to do, output raw scores
        return V
    }
    
    func backward(_ y: Int) -> Double {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = inAct else {
            fatalError("inAct is nil")
        }
        
        x.dw = ArrayUtils.zerosDouble(x.w.count) // zero out the gradient of input Vol
        
        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        let yscore = x.w[y] // score of ground truth
        let margin = 1.0
        var loss = 0.0
        for i in 0 ..< outDepth {
            
            if y == i { continue }
            let ydiff = -yscore + x.w[i] + margin
            if ydiff > 0 {
                // violating dimension, apply loss
                x.dw[i] += 1
                x.dw[y] -= 1
                loss += ydiff
            }
        }
        inAct = x
        return loss
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(_ paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = outDepth as AnyObject?
        json["outSx"] = outSx as AnyObject?
        json["outSy"] = outSy as AnyObject?
        json["layerType"] = layerType.rawValue as AnyObject?
        json["numInputs"] = numInputs as AnyObject?
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        outDepth = json["outDepth"]
    //        outSx = json["outSx"]
    //        outSy = json["outSy"]
    //        layerType = json["layerType"]
    //        numInputs = json["numInputs"]
    //    }
}


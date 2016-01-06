
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
    func backward(y: Int) -> Double
}

struct SoftmaxLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .Softmax
    
    var inSx: Int = 0
    var inSy: Int = 0
    var inDepth: Int = 0
    var num_classes: Int
    
    init (num_classes: Int) {
        self.num_classes = num_classes
    }
}

class SoftmaxLayer: LossLayer {
    
    var num_inputs: Int
    var outDepth: Int
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var inAct: Vol?
    var outAct: Vol?
    
    var es: [Double] = []
    
    init(opt: SoftmaxLayerOpt) {
        // computed
        self.num_inputs = opt.inSx * opt.inSy * opt.inDepth
        self.outDepth = self.num_inputs
        self.outSx = 1
        self.outSy = 1
        self.layerType = .Softmax
    }
    
    func forward(inout V: Vol, isTraining: Bool) -> Vol {
        self.inAct = V
        
        let A = Vol(sx: 1, sy: 1, depth: self.outDepth, c: 0.0)
        
        // compute max activation
        var a_s = V.w
        let amax = V.w.maxElement()!
        
        // compute exponentials (carefully to not blow up)
        var es = zerosd(self.outDepth)
        var esum = 0.0
        for i in 0 ..< self.outDepth {
            
            let e = exp(a_s[i] - amax)
            esum += e
            es[i] = e
        }
        
        // normalize and output to sum to one
        for i in 0 ..< self.outDepth {
            
            es[i] /= esum
            A.w[i] = es[i]
        }
        
        self.es = es // save these for backprop
        self.outAct = A
        return self.outAct!
    }
    
    func backward(y: Int) -> Double {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = self.inAct else {
            fatalError("self.inAct is nil")
        }
        x.dw = [Double](count: x.w.count, repeatedValue: 0.0) // zero out the gradient of input Vol
        
        for i in 0 ..< self.outDepth {
            
            let indicator = i == y ? 1.0 : 0.0
            let mul = -(indicator - self.es[i])
            x.dw[i] = mul
        }
        // loss is the class negative log likelihood
        return -log(self.es[y])
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        
        json["outDepth"] = self.outDepth
        json["outSx"] = self.outSx
        json["outSy"] = self.outSy
        json["layerType"] = self.layerType.rawValue
        json["num_inputs"] = self.num_inputs
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        self.outDepth = json["outDepth"]
    //        self.outSx = json["outSx"]
    //        self.outSy = json["outSy"]
    //        self.layerType = json["layerType"]
    //        self.num_inputs = json["num_inputs"]
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
    var num_neurons: Int
    
    init(num_neurons: Int) {
        // warning: creativity!
        self.inSx = 1
        self.inSy = 1
        self.inDepth = 1
        self.num_neurons = num_neurons
        
    }
}

class RegressionLayer: LossLayer {
    
    var num_inputs: Int
    var outDepth: Int
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: RegressionLayerOpt) {
        
        // computed
        self.num_inputs = opt.inSx * opt.inSy * opt.inDepth
        self.outDepth = self.num_inputs
        self.outSx = 1
        self.outSy = 1
        self.layerType = .Regression
    }

    func forward(inout V: Vol, isTraining: Bool) -> Vol {
        self.inAct = V
        self.outAct = V
        return V // identity function
    }

    // y is a list here of size num_inputs
    // or it can be a number if only one value is regressed
    // or it can be a struct {dim: i, val: x} where we only want to
    // regress on dimension i and asking it to have value x
    
    func backward(y: [Double]) -> Double {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = self.inAct else {
            fatalError("self.inAct is nil")
        }
        x.dw = [Double](count: x.w.count, repeatedValue: 0.0) // zero out the gradient of input Vol
        var loss = 0.0
        for i in 0 ..< self.outDepth {
            let dy = x.w[i] - y[i]
            x.dw[i] = dy
            loss += 0.5*dy*dy
        }
        self.inAct = x
        return loss
    }
    
    func backward(y: Double) -> Double {
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = self.inAct else {
            fatalError("self.inAct is nil")
        }
        x.dw = [Double](count: x.w.count, repeatedValue: 0.0) // zero out the gradient of input Vol
        var loss = 0.0
        // lets hope that only one number is being regressed
        let dy = x.w[0] - y
        x.dw[0] = dy
        loss += 0.5*dy*dy
        return loss
    }
    
    func backward(y: Int) -> Double {
        return backward(Double(y))
    }
    
    struct Pair {
        var dim: Int
        var val: Double
    }
    
    func backward(y: Pair) -> Double {
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = self.inAct else {
            fatalError("self.inAct is nil")
        }
        x.dw = [Double](count: x.w.count, repeatedValue: 0.0) // zero out the gradient of input Vol
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
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = self.outDepth
        json["outSx"] = self.outSx
        json["outSy"] = self.outSy
        json["layerType"] = self.layerType.rawValue
        json["num_inputs"] = self.num_inputs
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        self.outDepth = json["outDepth"]
    //        self.outSx = json["outSx"]
    //        self.outSy = json["outSy"]
    //        self.layerType = json["layerType"]
    //        self.num_inputs = json["num_inputs"]
    //    }
}

struct SVMLayerOpt: LayerInOptProtocol {
    var layerType: LayerType = .SVM
    
    var inSx: Int
    var inSy: Int
    var inDepth: Int
    var num_classes: Int
}

class SVMLayer: LossLayer {
    
    var num_inputs: Int
    var outDepth: Int
    var outSx: Int
    var outSy: Int
    var layerType: LayerType
    var inAct: Vol?
    var outAct: Vol?
    
    init(opt: SVMLayerOpt){
        // computed
        self.num_inputs = opt.inSx * opt.inSy * opt.inDepth
        self.outDepth = self.num_inputs
        self.outSx = 1
        self.outSy = 1
        self.layerType = .SVM
    }
    
    func forward(inout V: Vol, isTraining: Bool) -> Vol {
        self.inAct = V
        self.outAct = V // nothing to do, output raw scores
        return V
    }
    
    func backward(y: Int) -> Double {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = self.inAct else {
            fatalError("self.inAct is nil")
        }
        
        x.dw = [Double](count: x.w.count, repeatedValue: 0.0) // zero out the gradient of input Vol
        
        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        let yscore = x.w[y] // score of ground truth
        let margin = 1.0
        var loss = 0.0
        for i in 0 ..< self.outDepth {
            
            if(y == i) { continue }
            let ydiff = -yscore + x.w[i] + margin
            if(ydiff > 0) {
                // violating dimension, apply loss
                x.dw[i] += 1
                x.dw[y] -= 1
                loss += ydiff
            }
        }
        self.inAct = x
        return loss
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["outDepth"] = self.outDepth
        json["outSx"] = self.outSx
        json["outSy"] = self.outSy
        json["layerType"] = self.layerType.rawValue
        json["num_inputs"] = self.num_inputs
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        self.outDepth = json["outDepth"]
    //        self.outSx = json["outSx"]
    //        self.outSy = json["outSy"]
    //        self.layerType = json["layerType"]
    //        self.num_inputs = json["num_inputs"]
    //    }
}


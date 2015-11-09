
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
    var layer_type: LayerType = .softmax
    
    var in_sx: Int = 0
    var in_sy: Int = 0
    var in_depth: Int = 0
    var num_classes: Int
    
    init (num_classes: Int) {
        self.num_classes = num_classes
    }
}

class SoftmaxLayer: LossLayer {
    
    var num_inputs: Int
    var out_depth: Int
    var out_sx: Int
    var out_sy: Int
    var layer_type: LayerType
    var in_act: Vol?
    var out_act: Vol?
    
    var es: [Double] = []
    
    init(opt: SoftmaxLayerOpt) {
        // computed
        self.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth
        self.out_depth = self.num_inputs
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = .softmax
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        
        let A = Vol(1, 1, self.out_depth, 0.0)
        
        // compute max activation
        var a_s = V.w
        var amax = V.w[0]
        for(var i:Int = 1; i < self.out_depth; i++) {
            
            if(a_s[i] > amax) { amax = a_s[i] }
        }
        
        // compute exponentials (carefully to not blow up)
        var es = [Double](count: self.out_depth, repeatedValue: 0.0)
        var esum = 0.0
        for i in 0 ..< self.out_depth {
            
            let e = exp(a_s[i] - amax)
            esum += e
            es[i] = e
        }
        
        // normalize and output to sum to one
        for i in 0 ..< self.out_depth {
            
            es[i] /= esum
            A.w[i] = es[i]
        }
        
        self.es = es // save these for backprop
        self.out_act = A
        return self.out_act!
    }
    
    func backward(y: Int) -> Double {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = self.in_act else {
            fatalError("self.in_act is nil")
        }
        x.dw = [Double](count: x.w.count, repeatedValue: 0.0) // zero out the gradient of input Vol
        
        for i in 0 ..< self.out_depth {
            
            let indicator = i == y ? 1.0 : 0.0
            let mul = -(indicator - self.es[i])
            x.dw[i] = mul
        }
//        self.in_act = x
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
        
        json["out_depth"] = self.out_depth
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["layer_type"] = self.layer_type.rawValue
        json["num_inputs"] = self.num_inputs
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        self.out_depth = json["out_depth"]
    //        self.out_sx = json["out_sx"]
    //        self.out_sy = json["out_sy"]
    //        self.layer_type = json["layer_type"]
    //        self.num_inputs = json["num_inputs"]
    //    }
}

// implements an L2 regression cost layer,
// so penalizes \sum_i(||x_i - y_i||^2), where x is its input
// and y is the user-provided array of "correct" values.

struct RegressionLayerOpt: LayerInOptProtocol {
    var layer_type: LayerType = .regression
    
    var in_sx: Int
    var in_sy: Int
    var in_depth: Int
    var num_neurons: Int
    
    init(num_neurons: Int) {
        // warning: creativity!
        self.in_sx = 1
        self.in_sy = 1
        self.in_depth = 1
        self.num_neurons = num_neurons
        
    }
}

class RegressionLayer: LossLayer {
    
    var num_inputs: Int
    var out_depth: Int
    var out_sx: Int
    var out_sy: Int
    var layer_type: LayerType
    var in_act: Vol?
    var out_act: Vol?
    
    init(opt: RegressionLayerOpt) {
        
        // computed
        self.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth
        self.out_depth = self.num_inputs
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = .regression
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        self.out_act = V
        return V // identity function
    }
    
    // y is a list here of size num_inputs
    // or it can be a number if only one value is regressed
    // or it can be a struct {dim: i, val: x} where we only want to
    // regress on dimension i and asking it to have value x
    
    func backward(y: [Double]) -> Double {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = self.in_act else {
            fatalError("self.in_act is nil")
        }
        x.dw = [Double](count: x.w.count, repeatedValue: 0.0) // zero out the gradient of input Vol
        var loss = 0.0
        for i in 0 ..< self.out_depth {
            
            let dy = x.w[i] - y[i]
            x.dw[i] = dy
            loss += 0.5*dy*dy
        }
        self.in_act = x
        return loss
    }
    
    func backward(y: Double) -> Double {
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = self.in_act else {
            fatalError("self.in_act is nil")
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
        guard let x = self.in_act else {
            fatalError("self.in_act is nil")
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
        json["out_depth"] = self.out_depth
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["layer_type"] = self.layer_type.rawValue
        json["num_inputs"] = self.num_inputs
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        self.out_depth = json["out_depth"]
    //        self.out_sx = json["out_sx"]
    //        self.out_sy = json["out_sy"]
    //        self.layer_type = json["layer_type"]
    //        self.num_inputs = json["num_inputs"]
    //    }
}

struct SVMLayerOpt: LayerInOptProtocol {
    var layer_type: LayerType = .svm
    
    var in_sx: Int
    var in_sy: Int
    var in_depth: Int
    var num_classes: Int
}

class SVMLayer: LossLayer {
    
    var num_inputs: Int
    var out_depth: Int
    var out_sx: Int
    var out_sy: Int
    var layer_type: LayerType
    var in_act: Vol?
    var out_act: Vol?
    
    init(opt: SVMLayerOpt){
        // computed
        self.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth
        self.out_depth = self.num_inputs
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = .svm
    }
    
    func forward(inout V: Vol, is_training: Bool) -> Vol {
        self.in_act = V
        self.out_act = V // nothing to do, output raw scores
        return V
    }
    
    func backward(y: Int) -> Double {
        
        // compute and accumulate gradient wrt weights and bias of this layer
        guard let x = self.in_act else {
            fatalError("self.in_act is nil")
        }
        
        x.dw = [Double](count: x.w.count, repeatedValue: 0.0) // zero out the gradient of input Vol
        
        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        let yscore = x.w[y] // score of ground truth
        let margin = 1.0
        var loss = 0.0
        for i in 0 ..< self.out_depth {
            
            if(y == i) { continue }
            let ydiff = -yscore + x.w[i] + margin
            if(ydiff > 0) {
                // violating dimension, apply loss
                x.dw[i] += 1
                x.dw[y] -= 1
                loss += ydiff
            }
        }
        self.in_act = x
        return loss
    }
    
    func getParamsAndGrads() -> [ParamsAndGrads] {
        return []
    }
    
    func assignParamsAndGrads(paramsAndGrads: [ParamsAndGrads]) {
        
    }
    
    func toJSON() -> [String: AnyObject] {
        var json: [String: AnyObject] = [:]
        json["out_depth"] = self.out_depth
        json["out_sx"] = self.out_sx
        json["out_sy"] = self.out_sy
        json["layer_type"] = self.layer_type.rawValue
        json["num_inputs"] = self.num_inputs
        return json
    }
    //
    //    func fromJSON(json: [String: AnyObject]) -> () {
    //        self.out_depth = json["out_depth"]
    //        self.out_sx = json["out_sx"]
    //        self.out_sy = json["out_sy"]
    //        self.layer_type = json["layer_type"]
    //        self.num_inputs = json["num_inputs"]
    //    }
}


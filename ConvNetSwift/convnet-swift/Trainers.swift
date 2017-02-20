import Foundation

enum TrainerType: String {
    case sgd = "sgd"
    case adam = "adam"
    case adagrad = "adagrad"
    case windowgrad = "windowgrad"
    case adadelta = "adadelta"
    case nesterov = "nesterov"
}

struct TrainerOpt {
    var method: TrainerType = .sgd
    var batchSize: Int = 1
    var l1Decay: Double = 0.0
    var l2Decay: Double = 0.0
    var learningRate: Double = 0.01
    var momentum: Double = 0.9
    var ρ: Double = 0.95
    var ε: Double = 1e-8
    var β1: Double = 0.9
    var β2: Double = 0.999
}

class Trainer {
    
    struct TrainResult {
        var forwardTime: Int
        var backwardTime: Int
        var l2DecayLoss: Double
        var l1DecayLoss: Double
        var costLoss: Double
        var softmaxLoss: Double
        var loss: Double
    }
    
    var net: Net
    var learningRate: Double = 0.0
    var l1Decay: Double = 0.0
    var l2Decay: Double = 0.0
    var batchSize: Int = 0
    var method: TrainerType
    var momentum: Double = 0.0
    var ρ: Double = 0.0
    var ε: Double = 0.0
    var β1: Double = 0.0
    var β2: Double = 0.0
    var k: Int = 0
    var gsum: [[Double]] = []
    var xsum: [[Double]] = []
    var regression: Bool = false
    
    init(net: Net, options: TrainerOpt) {
        
        self.net = net
        
        learningRate = options.learningRate 
        l1Decay = options.l1Decay
        l2Decay = options.l2Decay
        batchSize = options.batchSize
        method = options.method // sgd/adam/adagrad/adadelta/windowgrad/netsterov
        
        momentum = options.momentum
        ρ = options.ρ // used in adadelta
        ε = options.ε // used in adam or adadelta
        β1 = options.β1 // used in adam
        β2 = options.β2 // used in adam
        
        k = 0 // iteration counter
        gsum = [] // last iteration gradients (used for momentum calculations)
        xsum = [] // used in adam or adadelta
        
        // check if regression is expected
        if net.layers[net.layers.count - 1].layerType == .Regression {
            regression = true
        } else {
            regression = false
        }
    }
    
    func train(x: inout Vol, y: [Double]) -> TrainResult {
        assert(regression)
        
        let startf = Date()
        _ = net.forward(&x, isTraining: true) // also set the flag that lets the net know we're just training
        let endf = Date()
        let forwardTime = endf.nanoseconds(from: startf)
        
        let startb = Date()
        let costLoss = net.backward(y)
        let endb = Date()
        let backwardTime = endb.nanoseconds(from: startb)
        
        let (l1DecayLoss, l2DecayLoss) = _perform_train()
        
        // appending softmaxLoss for backwards compatibility, but from now on we will always use costLoss
        // in future, TODO: have to completely redo the way loss is done around the network as currently
        // loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
        // and it should all be computed correctly and automatically.
        return TrainResult(
            forwardTime: forwardTime,
            backwardTime: backwardTime,
            l2DecayLoss: l2DecayLoss,
            l1DecayLoss: l1DecayLoss,
            costLoss: costLoss,
            softmaxLoss: costLoss,
            loss: costLoss + l1DecayLoss + l2DecayLoss)
    }
    
    func train(x: inout Vol, y: RegressionLayer.Pair) -> TrainResult {
        assert(regression)
        
        let startf = Date()
        _ = net.forward(&x, isTraining: true) // also set the flag that lets the net know we're just training
        let endf = Date()
        let forwardTime = endf.nanoseconds(from: startf)
        
        let startb = Date()
        let costLoss = net.backward(y)
        let endb = Date()
        let backwardTime = endb.nanoseconds(from: startb)
        
        let (l1DecayLoss, l2DecayLoss) = _perform_train()
        
        return TrainResult(
            forwardTime: forwardTime,
            backwardTime: backwardTime,
            l2DecayLoss: l2DecayLoss,
            l1DecayLoss: l1DecayLoss,
            costLoss: costLoss,
            softmaxLoss: costLoss,
            loss: costLoss + l1DecayLoss + l2DecayLoss)
    }
    
    func train(x: inout Vol, y: Int) -> TrainResult {
        assert(!regression, "y should be an array if you want to do a regression.")
        
        let startf = Date()
        _ = net.forward(&x, isTraining: true) // also set the flag that lets the net know we're just training
        let endf = Date()
        let forwardTime = endf.nanoseconds(from: startf)
        
        let startb = Date()
        let costLoss = net.backward(y)
        let endb = Date()
        let backwardTime = endb.nanoseconds(from: startb)
        
        let (l1DecayLoss, l2DecayLoss) = _perform_train()
        
        return TrainResult(
            forwardTime: forwardTime,
            backwardTime: backwardTime,
            l2DecayLoss: l2DecayLoss,
            l1DecayLoss: l1DecayLoss,
            costLoss: costLoss,
            softmaxLoss: costLoss,
            loss: costLoss + l1DecayLoss + l2DecayLoss)
    }
    
    func _perform_train() -> (l1DecayLoss: Double, l2DecayLoss: Double) {
        var l2DecayLoss = 0.0
        var l1DecayLoss = 0.0
        
        k += 1
        if k % batchSize == 0 {
            
            var pglist = net.getParamsAndGrads()
            var newParamsAndGradients: [ParamsAndGrads] = []
            
            // initialize lists for accumulators. Will only be done once on first iteration
            if gsum.count == 0 && (method != .sgd || momentum > 0.0) {
                // only vanilla sgd doesnt need either lists
                // momentum needs gsum
                // adagrad needs gsum
                // adam and adadelta needs gsum and xsum
                for i in 0 ..< pglist.count {
                    
                    gsum.append(ArrayUtils.zerosDouble(pglist[i].params.count))
                    if method == TrainerType.adam || method == TrainerType.adadelta {
                        xsum.append(ArrayUtils.zerosDouble(pglist[i].params.count))
                    } else {
                        xsum.append([]) // conserve memory
                    }
                }
            }
            
            // perform an update for all sets of weights
            for i in 0 ..< pglist.count {
                
                let pg = pglist[i] // param, gradient, other options in future (custom learning rate etc)
                var p = pg.params
                var g = pg.grads
                
                // learning rate for some parameters.
                let l2DecayMul = pg.l2DecayMul ?? 1.0
                let l1DecayMul = pg.l1DecayMul ?? 1.0
                let l2Decay = self.l2Decay * l2DecayMul
                let l1Decay = self.l1Decay * l1DecayMul
                
                let plen = p.count
                for j in 0 ..< plen {
                    
                    l2DecayLoss += l2Decay*p[j]*p[j]/2 // accumulate weight decay loss
                    l1DecayLoss += l1Decay*abs(p[j])
                    let l1grad = l1Decay * (p[j] > 0 ? 1 : -1)
                    let l2grad = l2Decay * (p[j])
                    
                    let gij = (l2grad + l1grad + g[j]) / Double(batchSize) // raw batch gradient
                    
                    var gsumi: [Double] = []
                    var xsumi: [Double] = []
                    
                    if method != .sgd || momentum > 0.0 {
                        gsumi = gsum[i]
                        xsumi = xsum[i]
                    }
                    
                    if method == .adam {
                        // adam update
                        gsumi[j] = gsumi[j] * β1 + (1-β1) * gij // update biased first moment estimate
                        xsumi[j] = xsumi[j] * β2 + (1-β2) * gij * gij // update biased second moment estimate
                        let biasCorr1 = gsumi[j] / (1 - pow(β1, Double(k))) // correct bias first moment estimate
                        let biasCorr2 = xsumi[j] / (1 - pow(β2, Double(k))) // correct bias second moment estimate
                        let dx =  -learningRate * biasCorr1 / (sqrt(biasCorr2) + ε)
                        p[j] += dx
                    } else if method == .adagrad {
                        // adagrad update
                        gsumi[j] = gsumi[j] + gij * gij
                        let dx = -learningRate / sqrt(gsumi[j] + ε) * gij
                        p[j] += dx
                    } else if method == .windowgrad {
                        // this is adagrad but with a moving window weighted average
                        // so the gradient is not accumulated over the entire history of the run.
                        // it's also referred to as Idea #1 in Zeiler paper on Adadelta. Seems reasonable to me!
                        gsumi[j] = ρ * gsumi[j] + (1-ρ) * gij * gij
                        let dx = -learningRate / sqrt(gsumi[j] + ε) * gij // eps added for better conditioning
                        p[j] += dx
                    } else if method == .adadelta {
                        gsumi[j] = ρ * gsumi[j] + (1-ρ) * gij * gij
                        let dx = -sqrt((xsumi[j] + ε)/(gsumi[j] + ε)) * gij
                        xsumi[j] = ρ * xsumi[j] + (1-ρ) * dx * dx // yes, xsum lags behind gsum by 1.
                        p[j] += dx
                    } else if method == .nesterov {
                        var dx = gsumi[j]
                        gsumi[j] = gsumi[j] * momentum + learningRate * gij
                        dx = momentum * dx - (1.0 + momentum) * gsumi[j]
                        p[j] += dx
                    } else {
                        // assume SGD
                        if momentum > 0.0 {
                            // momentum update
                            let dx = momentum * gsumi[j] - learningRate * gij // step
                            gsumi[j] = dx // back this up for next iteration of momentum
                            p[j] += dx // apply corrected gradient
                        } else {
                            // vanilla sgd
                            p[j] +=  -learningRate * gij
                        }
                    }
                    g[j] = 0.0 // zero out gradient so that we can begin accumulating anew
                }
                
                newParamsAndGradients.append(
                    ParamsAndGrads(params: &p, grads: &g, l1DecayMul: l1DecayMul, l2DecayMul: l2DecayMul)
                )
            }
            
            net.assignParamsAndGrads(newParamsAndGradients)
        }
        return (l1DecayLoss, l2DecayLoss)
    }
    
}


import Foundation
import GameplayKit


// Random number utilities

class RandUtils {
    
    static var return_v = false
    static var v_val = 0.0
    
    static func random_js() -> Double {
        return drand48()
        // should be [0 .. 1)
    }
    
    static func randf(_ a: Double, _ b: Double) -> Double {
        return random_js()*(b-a)+a
    }
    
    static func randi(_ a: Int, _ b: Int) -> Int {
        return Int(floor(random_js()))*(b-a)+a
    }
    
    private static let gaussDistribution = GKGaussianDistribution(randomSource: GKRandomSource(), mean:0, deviation: 1)
    
    static func randn(_ mu: Double, std: Double) -> Double {
        return (Double(gaussDistribution.nextUniform()) + mu) * std
    }
    
}
// Array utilities

struct ArrayUtils {
    static func zerosInt(_ n: Int) -> [Int] {
        return [Int](repeating: 0, count: n)
    }
    
    static func zerosDouble(_ n: Int) -> [Double] {
        return [Double](repeating: 0.0, count: n)
    }
    
    static func zerosBool(_ n: Int) -> [Bool] {
        return [Bool](repeating: false, count: n)
    }
    
    static func arrUnique(_ arr: [Int]) -> [Int] {
        return Array(Set(arr))
    }
}

// return max and min of a given non-empty array.
struct Maxmin {
    var maxi: Int
    var maxv: Double
    var mini: Int
    var minv: Double
    var dv: Double
}

func maxmin(_ w: [Double]) -> Maxmin? {
    guard (w.count > 0),
        let maxv = w.max(),
        let maxi = w.index(of: maxv),
        let minv = w.min(),
        let mini = w.index(of: minv)
        else {
            return nil
    }
    return Maxmin(maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv: maxv-minv)
}

// create random permutation of numbers, in range [0...n-1]
func randomPermutation(_ n: Int) -> [Int]{
    let dist = GKShuffledDistribution(lowestValue: 0, highestValue: n-1)
    return (0..<n).map{_ in dist.nextInt()}
}

// sample from list lst according to probabilities in list probs
// the two lists are of same size, and probs adds up to 1
func weightedSample(_ lst: [Double], probs: [Double]) -> Double? {
    let p = RandUtils.randf(0, 1.0)
    var cumprob = 0.0
    let n=lst.count
    for k in 0 ..< n {
        cumprob += probs[k]
        if p < cumprob { return lst[k] }
    }
    return nil
}

struct TimeUtils {
    static func getNanoseconds() {
        
    }
}


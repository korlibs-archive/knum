package com.soywiz.knum.tensorflow

import com.soywiz.knum.KNum
import com.soywiz.knum.KNumContext

fun <T> KNumTensorFlow(callback: KNum.() -> T) = KNum({ TensorFlowKNumContext() }, callback)

class TensorFlowKNumContext : KNumContext() {

}

package com.soywiz.knum.opencl

import com.soywiz.knum.KNum
import com.soywiz.knum.KNumContext

fun <T> KNumOpenCl(callback: KNum.() -> T) = KNum(OpenClKNumContext(), callback)

class OpenClKNumContext : KNumContext() {
    override fun <T> session(callback: () -> T): T {
        return callback()
    }
}

object KNumOpenClExample {
    @JvmStatic
    fun main(args: Array<String>) {
        KNumOpenCl {
        }
    }
}

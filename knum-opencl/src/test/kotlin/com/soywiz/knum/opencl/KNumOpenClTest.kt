package com.soywiz.knum.opencl

import com.soywiz.knum.KNumContext
import com.soywiz.knum.KNumTest

class KNumOpenClTest : KNumTest() {
    override fun contextGenerate(): KNumContext = KNumOpenClContext()
}
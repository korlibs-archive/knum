package com.soywiz.knum

import java.nio.Buffer
import java.nio.FloatBuffer
import java.nio.IntBuffer

private inline fun <T> Buffer.keepPosition(callback: () -> T): T {
    val oldPosition = position()
    return callback().apply { position(oldPosition) }
}

fun IntBuffer.toTypedArray(): IntArray = IntArray(limit()).apply { keepPosition { this@toTypedArray.get(this) } }
fun FloatBuffer.toTypedArray(): FloatArray = FloatArray(limit()).apply { keepPosition { this@toTypedArray.get(this) } }

fun IntBuffer.toList() = toTypedArray().toList()
fun FloatBuffer.toList() = toTypedArray().toList()

package com.soywiz.knum.opencl

import org.intellij.lang.annotations.Language
import org.jocl.*
import org.jocl.CL.*
import java.io.Closeable
import java.nio.FloatBuffer


class ClContext : Closeable {
    val context: cl_context
    val platformIndex = 0
    val deviceType = CL_DEVICE_TYPE_ALL
    val deviceIndex = 0
    val device: cl_device_id

    init {
        setExceptionsEnabled(true)

        // Obtain the number of platforms
        val numPlatformsArray = IntArray(1)
        clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        // Obtain a platform ID
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        clGetPlatformIDs(platforms.size, platforms, null)
        val platform = platforms[platformIndex]

        // Initialize the context properties
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(CL_CONTEXT_PLATFORM.toLong(), platform)

        // Obtain the number of devices for the platform
        val numDevicesArray = IntArray(1)
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]

        // Obtain a device ID
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        device = devices[deviceIndex]!!

        // Create a context for the selected device
        context = clCreateContext(contextProperties, 1, arrayOf(device), null, null, null)
    }

    fun createCommandQueue() = ClCommandQueue(this)
    fun createBuffer(data: FloatArray, size: Int = data.size, writeable: Boolean = false) = ClBuffer(this, Pointer.to(data), Sizeof.cl_float, size, writeable)
    fun createBuffer(data: FloatBuffer, size: Int = data.limit(), writeable: Boolean = false) = ClBuffer(this, Pointer.to(data), Sizeof.cl_float, size, writeable)
    fun createEmptyBuffer(elementSize: Int, length: Int, writeable: Boolean = true) = ClBuffer(this, null, elementSize, length, writeable = true)
    fun createProgram(@Language("opencl") source: String) = ClProgram(this, source)

    inline fun <T> queue(callback: ClQueueContext.() -> T): T {
        return createCommandQueue().use { queue ->
            callback(ClQueueContext(queue))
        }
    }

    override fun close() {
        clReleaseContext(context)
    }
}

class ClQueueContext(val queue: ClCommandQueue) {
    fun ClBuffer.readFloats() = readFloats(queue)
    fun ClBuffer.readFloatsQueue() = readFloatsQueue(queue)

    operator fun ClKernel.invoke(vararg args: ClBuffer) = invoke(queue, *args)
    fun ClKernel.invokeQueue(vararg args: ClBuffer) = invokeQueue(queue, *args)
}

class ClBuffer(val ctx: ClContext, val ptr: Pointer?, val elementSize: Int, val length: Int, val writeable: Boolean) : Closeable {
    val sizeInBytes: Int = elementSize * length
    private val flags = run {
        var flags = 0L
        flags = flags or (if (writeable) CL_MEM_READ_WRITE else CL_MEM_READ_ONLY)
        flags = flags or (if (ptr != null) CL_MEM_COPY_HOST_PTR else 0L)
        flags
    }
    val mem = clCreateBuffer(ctx.context, flags, sizeInBytes.toLong(), ptr, null)

    fun readFloats(queue: ClCommandQueue): FloatArray = queue.readFloats(this).apply { queue.waitCompleted() }
    fun readFloatsQueue(queue: ClCommandQueue): FloatArray = queue.readFloats(this)

    override fun close() {
        clReleaseMemObject(mem)
    }
}

class ClProgram(val ctx: ClContext, val source: String) : Closeable {
    val result = IntArray(1)
    val program = clCreateProgramWithSource(ctx.context, 1, arrayOf(source), longArrayOf(source.length.toLong()), result)

    init {
        clBuildProgram(program, 0, null, null, null, null)
    }

    fun getKernel(name: String) = ClKernel(this, name)

    override fun close() {
        clReleaseProgram(program)
    }
}

class ClKernel(val program: ClProgram, val name: String) {
    operator fun invoke(queue: ClCommandQueue, vararg args: ClBuffer) {
        invokeQueue(queue, *args)
        queue.waitCompleted()
    }

    fun invokeQueue(queue: ClCommandQueue, vararg args: ClBuffer) {
        val errorCode = IntArray(1)
        val kernel = clCreateKernel(program.program, name, errorCode)
        for ((index, arg) in args.withIndex()) {
            clSetKernelArg(kernel, index, Sizeof.cl_mem.toLong(), Pointer.to(arg.mem))
        }
        val global_work_size = longArrayOf(args[0].length.toLong())
        val local_work_size = longArrayOf(1)
        clEnqueueNDRangeKernel(queue.commandQueue, kernel, 1, null, global_work_size, local_work_size, 0, null, null)
        clReleaseKernel(kernel)
    }
}

class ClCommandQueue(val ctx: ClContext) : Closeable {
    val commandQueue = clCreateCommandQueue(ctx.context, ctx.device, 0, null)

    fun readFloats(buffer: ClBuffer): FloatArray {
        val out = FloatArray(buffer.length)
        clEnqueueReadBuffer(commandQueue, buffer.mem, CL_TRUE, 0, buffer.sizeInBytes.toLong(), Pointer.to(out), 0, null, null)
        return out
    }

    fun waitCompleted() {
        clFinish(commandQueue)
    }

    override fun close() {
        waitCompleted()
        clReleaseCommandQueue(commandQueue)
    }
}

object ClExample {
    @JvmStatic
    fun main(args: Array<String>) {
        ClContext().run {
            queue {
                val program = createProgram("""
                    __kernel void sampleKernel(__global const float *a, __global const float *b, __global float *c) {
                        int gid = get_global_id(0);
                        c[gid] = a[gid] * b[gid] * 2.0;
                    }
                """)

                val buffer1 = createBuffer(floatArrayOf(1f, 2f, 3f, 4f))
                val buffer2 = createBuffer(floatArrayOf(5f, 6f, 7f, 8f))
                val buffer3 = createEmptyBuffer(4, 4, writeable = true)
                val sampleKernel = program.getKernel("sampleKernel")
                sampleKernel(buffer1, buffer2, buffer3)

                val result = buffer3.readFloats(queue)

                println(result.toList())

                buffer3.close()
                buffer2.close()
                buffer1.close()
            }
        }
    }
}

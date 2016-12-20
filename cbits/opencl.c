#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

#include <stdio.h>

#define CONSTANT_FUN(prefix) \
    cl_uint cl_##prefix( void ) \
    { return CL_##prefix; }

CONSTANT_FUN(INVALID_LINKER_OPTIONS)
CONSTANT_FUN(LINK_PROGRAM_FAILURE)
CONSTANT_FUN(INVALID_COMPILER_OPTIONS)
CONSTANT_FUN(BUILD_PROGRAM_FAILURE)
CONSTANT_FUN(COMPILE_PROGRAM_FAILURE)
CONSTANT_FUN(PROGRAM_BUILD_LOG)
CONSTANT_FUN(MAP_READ)
CONSTANT_FUN(MAP_WRITE)
CONSTANT_FUN(MAP_WRITE_INVALIDATE_REGION)
CONSTANT_FUN(MEM_READ_WRITE)
CONSTANT_FUN(MEM_READ_ONLY)
CONSTANT_FUN(MEM_WRITE_ONLY)
CONSTANT_FUN(MEM_USE_HOST_PTR)
CONSTANT_FUN(MEM_ALLOC_HOST_PTR)
CONSTANT_FUN(MEM_COPY_HOST_PTR)
CONSTANT_FUN(MEM_HOST_WRITE_ONLY)
CONSTANT_FUN(MEM_HOST_READ_ONLY)
CONSTANT_FUN(MEM_HOST_NO_ACCESS)
CONSTANT_FUN(EXEC_KERNEL);
CONSTANT_FUN(EXEC_NATIVE_KERNEL);
CONSTANT_FUN(READ_ONLY_CACHE);
CONSTANT_FUN(READ_WRITE_CACHE);
CONSTANT_FUN(LOCAL);
CONSTANT_FUN(GLOBAL);
CONSTANT_FUN(DEVICE_TYPE_CPU);
CONSTANT_FUN(DEVICE_TYPE_GPU);
CONSTANT_FUN(DEVICE_TYPE_ACCELERATOR);
CONSTANT_FUN(DEVICE_TYPE_DEFAULT);
CONSTANT_FUN(DEVICE_TYPE_CUSTOM);
CONSTANT_FUN(DEVICE_PARTITION_TYPE);
CONSTANT_FUN(DEVICE_PARTITION_PROPERTIES);
CONSTANT_FUN(DEVICE_PARTITION_EQUALLY);
CONSTANT_FUN(DEVICE_PARTITION_BY_COUNTS);
CONSTANT_FUN(DEVICE_PARTITION_BY_AFFINITY_DOMAIN);
CONSTANT_FUN(DEVICE_AFFINITY_DOMAIN_NUMA);
CONSTANT_FUN(DEVICE_AFFINITY_DOMAIN_L4_CACHE);
CONSTANT_FUN(DEVICE_AFFINITY_DOMAIN_L3_CACHE);
CONSTANT_FUN(DEVICE_AFFINITY_DOMAIN_L2_CACHE);
CONSTANT_FUN(DEVICE_AFFINITY_DOMAIN_L1_CACHE);
CONSTANT_FUN(DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE);
CONSTANT_FUN(QUEUE_PROFILING_ENABLE);
CONSTANT_FUN(QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
CONSTANT_FUN(FP_INF_NAN);
CONSTANT_FUN(FP_SOFT_FLOAT);
CONSTANT_FUN(FP_CORRECTLY_ROUNDED_DIVIDE_SQRT);
CONSTANT_FUN(FP_DENORM);
CONSTANT_FUN(FP_ROUND_TO_NEAREST);
CONSTANT_FUN(FP_ROUND_TO_INF);
CONSTANT_FUN(FP_ROUND_TO_ZERO);
CONSTANT_FUN(FP_FMA);
CONSTANT_FUN(PLATFORM_PROFILE);
CONSTANT_FUN(PLATFORM_NAME);
CONSTANT_FUN(PLATFORM_VERSION);
CONSTANT_FUN(PLATFORM_VENDOR);
CONSTANT_FUN(PLATFORM_EXTENSIONS);
CONSTANT_FUN(DEVICE_EXTENSIONS);
CONSTANT_FUN(DEVICE_MAX_CLOCK_FREQUENCY);
CONSTANT_FUN(DEVICE_PARENT_DEVICE);
CONSTANT_FUN(DEVICE_PLATFORM);
CONSTANT_FUN(DEVICE_MAX_READ_IMAGE_ARGS);
CONSTANT_FUN(DEVICE_IMAGE_MAX_ARRAY_SIZE);
CONSTANT_FUN(DEVICE_IMAGE_MAX_BUFFER_SIZE);
CONSTANT_FUN(DEVICE_ERROR_CORRECTION_SUPPORT);
CONSTANT_FUN(DEVICE_ENDIAN_LITTLE);
CONSTANT_FUN(DEVICE_EXECUTION_CAPABILITIES);
CONSTANT_FUN(DEVICE_GLOBAL_MEM_CACHE_SIZE);
CONSTANT_FUN(DEVICE_GLOBAL_MEM_CACHE_TYPE);
CONSTANT_FUN(DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
CONSTANT_FUN(DEVICE_GLOBAL_MEM_SIZE);
CONSTANT_FUN(DEVICE_QUEUE_PROPERTIES);
CONSTANT_FUN(DEVICE_PARTITION_AFFINITY_DOMAIN);
CONSTANT_FUN(DEVICE_MAX_WORK_ITEM_DIMENSIONS);
CONSTANT_FUN(DEVICE_MAX_WORK_ITEM_SIZES);
CONSTANT_FUN(DEVICE_MAX_WRITE_IMAGE_ARGS);
CONSTANT_FUN(DEVICE_MEM_BASE_ADDR_ALIGN);
CONSTANT_FUN(DEVICE_MIN_DATA_TYPE_ALIGN_SIZE);
CONSTANT_FUN(DEVICE_MAX_CONSTANT_ARGS);
CONSTANT_FUN(DEVICE_MAX_CONSTANT_BUFFER_SIZE);
CONSTANT_FUN(DEVICE_MAX_MEM_ALLOC_SIZE);
CONSTANT_FUN(DEVICE_LOCAL_MEM_SIZE);
CONSTANT_FUN(DEVICE_LOCAL_MEM_TYPE);
CONSTANT_FUN(DEVICE_COMPILER_AVAILABLE);
CONSTANT_FUN(DEVICE_LINKER_AVAILABLE);
CONSTANT_FUN(DEVICE_PARTITION_MAX_SUB_DEVICES);
CONSTANT_FUN(DEVICE_PREFERRED_INTEROP_USER_SYNC);
CONSTANT_FUN(DEVICE_ADDRESS_BITS);
CONSTANT_FUN(DEVICE_AVAILABLE);
CONSTANT_FUN(DEVICE_IMAGE2D_MAX_HEIGHT);
CONSTANT_FUN(DEVICE_IMAGE2D_MAX_WIDTH);
CONSTANT_FUN(DEVICE_IMAGE3D_MAX_DEPTH);
CONSTANT_FUN(DEVICE_IMAGE3D_MAX_HEIGHT);
CONSTANT_FUN(DEVICE_IMAGE3D_MAX_WIDTH);
CONSTANT_FUN(DEVICE_MAX_SAMPLERS);
CONSTANT_FUN(DEVICE_MAX_COMPUTE_UNITS);
CONSTANT_FUN(DEVICE_MAX_PARAMETER_SIZE);
CONSTANT_FUN(DEVICE_MAX_WORK_GROUP_SIZE);
CONSTANT_FUN(DEVICE_BUILT_IN_KERNELS);
CONSTANT_FUN(DEVICE_HOST_UNIFIED_MEMORY);
CONSTANT_FUN(DEVICE_IMAGE_SUPPORT);
CONSTANT_FUN(DEVICE_NAME);
CONSTANT_FUN(DEVICE_TYPE);
CONSTANT_FUN(DEVICE_VENDOR_ID);
CONSTANT_FUN(DEVICE_OPENCL_C_VERSION);
CONSTANT_FUN(DEVICE_PROFILE);
CONSTANT_FUN(DEVICE_VENDOR);
CONSTANT_FUN(DEVICE_VERSION);
CONSTANT_FUN(DRIVER_VERSION);
CONSTANT_FUN(DEVICE_SINGLE_FP_CONFIG);
CONSTANT_FUN(DEVICE_DOUBLE_FP_CONFIG);
CONSTANT_FUN(DEVICE_PROFILING_TIMER_RESOLUTION);
CONSTANT_FUN(DEVICE_PRINTF_BUFFER_SIZE);
CONSTANT_FUN(DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
CONSTANT_FUN(DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
CONSTANT_FUN(DEVICE_PREFERRED_VECTOR_WIDTH_INT);
CONSTANT_FUN(DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
CONSTANT_FUN(DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
CONSTANT_FUN(DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
CONSTANT_FUN(DEVICE_PREFERRED_VECTOR_WIDTH_HALF);
CONSTANT_FUN(DEVICE_NATIVE_VECTOR_WIDTH_CHAR);
CONSTANT_FUN(DEVICE_NATIVE_VECTOR_WIDTH_SHORT);
CONSTANT_FUN(DEVICE_NATIVE_VECTOR_WIDTH_INT);
CONSTANT_FUN(DEVICE_NATIVE_VECTOR_WIDTH_LONG);
CONSTANT_FUN(DEVICE_NATIVE_VECTOR_WIDTH_FLOAT);
CONSTANT_FUN(DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE);
CONSTANT_FUN(DEVICE_NATIVE_VECTOR_WIDTH_HALF);

cl_uint cl_DEVICE_HALF_FP_CONFIG( void )
{
    return 0x1033;
}

cl_int code_success( void )
{
    return CL_SUCCESS;
}

cl_int get_num_platforms( cl_uint* platforms )
{
    return clGetPlatformIDs( 0, NULL, platforms );
}

cl_int get_num_devices( cl_platform_id platform
                      , cl_uint* devices )
{
    return clGetDeviceIDs( platform
                         , CL_DEVICE_TYPE_ALL
                         , 0
                         , NULL
                         , devices );
}

cl_int get_devices( cl_platform_id platform
                  , cl_uint num_devices
                  , cl_device_id* devices )
{
    return clGetDeviceIDs( platform
                         , CL_DEVICE_TYPE_ALL
                         , num_devices
                         , devices
                         , NULL );
}

cl_int get_device_info_size( cl_device_id device
                           , cl_device_info param_name
                           , size_t* sz )
{
    return clGetDeviceInfo( device
                          , param_name
                          , 0
                          , NULL
                          , sz );
}

cl_int get_device_info( cl_device_id device
                      , cl_device_info param_name
                      , size_t sz
                      , void* out )
{
    return clGetDeviceInfo( device
                          , param_name
                          , sz
                          , out
                          , NULL );
}

cl_int get_platforms( cl_uint num_platforms, cl_platform_id* platforms )
{
    return clGetPlatformIDs( num_platforms, platforms, NULL );
}

cl_int get_platform_info_size( cl_platform_id platform
                             , cl_platform_info param_name
                             , size_t* sz )
{
    return clGetPlatformInfo( platform
                            , param_name
                            , 0
                            , NULL
                            , sz );
}

cl_int get_platform_info( cl_platform_id platform
                        , cl_platform_info param_name
                        , size_t param_sz
                        , void* out )
{
    return clGetPlatformInfo( platform
                            , param_name
                            , param_sz
                            , out
                            , NULL );
}

void release_device( cl_device_id device )
{
    clReleaseDevice( device );
}

void retain_context( cl_context context )
{
    clRetainContext( context );
}

void release_context( cl_context context )
{
    clReleaseContext( context );
}

void retain_device( cl_device_id device )
{
    clRetainDevice( device );
}

void retain_command_queue( cl_command_queue queue )
{
    clRetainCommandQueue( queue );
}

void release_command_queue( cl_command_queue queue )
{
    clReleaseCommandQueue( queue );
}

void release_mem( cl_mem mem )
{
    clReleaseMemObject( mem );
}

void retain_mem( cl_mem mem )
{
    clRetainMemObject( mem );
}

void release_event( cl_event ev )
{
    clReleaseEvent( ev );
}

void release_program( cl_program program )
{
    clReleaseProgram( program );
}

void release_kernel( cl_kernel kernel )
{
    clReleaseKernel( kernel );
}

cl_int enqueue_range_kernel( cl_command_queue queue
                           , cl_kernel kernel
                           , cl_uint work_dim
                           , const size_t* global_work_offset
                           , const size_t* global_work_size
                           , const size_t* local_work_size
                           , cl_uint num_events
                           , const cl_event* wait_list
                           , cl_event* event )
{
    return clEnqueueNDRangeKernel( queue, kernel, work_dim, global_work_offset, global_work_size, local_work_size, num_events, wait_list, event );
}

cl_kernel create_kernel( cl_program program
                       , const char* kernel_name
                       , cl_int* errcode_ret )
{
    return clCreateKernel( program, kernel_name, errcode_ret );
}

cl_int set_kernel_arg( cl_kernel kernel
                     , cl_uint arg_index
                     , size_t arg_size
                     , const void* arg_value )
{
    return clSetKernelArg( kernel, arg_index, arg_size, arg_value );
}

cl_program create_program_with_source( cl_context ctx
                                     , cl_uint count
                                     , const char** strings
                                     , const size_t* lengths
                                     , cl_int* errcode_ret )
{
    return clCreateProgramWithSource( ctx, count, strings, lengths, errcode_ret );
}

cl_int get_program_build_info( cl_program program
                             , cl_device_id device
                             , cl_program_build_info param_name
                             , size_t param_size
                             , void* param_value
                             , size_t* param_value_size_ret )
{
    return clGetProgramBuildInfo( program, device, param_name, param_size, param_value, param_value_size_ret );
}

cl_int compile_program( cl_program program
                      , cl_uint num_devices
                      , const cl_device_id* device_list
                      , const char* options
                      , cl_uint num_input_headers
                      , const cl_program* headers
                      , const char** header_include_names
                      , void (CL_CALLBACK *pfn_notify)(cl_program, void*)
                      , void* user_data )
{
    return clCompileProgram( program
                           , num_devices
                           , device_list
                           , options
                           , num_input_headers
                           , headers
                           , header_include_names
                           , pfn_notify
                           , user_data );
}

cl_program link_program( cl_context context
                       , cl_uint num_devices
                       , const cl_device_id* device_list
                       , const char* options
                       , cl_uint num_input_programs
                       , const cl_program* programs
                       , void (CL_CALLBACK *pfn_notify)(cl_program, void*)
                       , void* user_data
                       , cl_int* errcode_ret )
{
    return clLinkProgram( context, num_devices, device_list, options, num_input_programs, programs, pfn_notify, user_data, errcode_ret );
}

cl_int create_sub_devices( cl_device_id device
                         , const cl_device_partition_property* props
                         , cl_uint num_devices
                         , cl_device_id* out_devices
                         , cl_uint* num_devices_ret )
{
    return clCreateSubDevices( device
                             , props
                             , num_devices
                             , out_devices
                             , num_devices_ret );
}

cl_int get_buffer_size( cl_mem memobj, size_t* output )
{
    (*output) = 0;
    size_t out_size = 0;
    cl_int result = clGetMemObjectInfo( memobj
                                      , CL_MEM_SIZE
                                      , sizeof(size_t)
                                      , &out_size
                                      , NULL );
    if ( result != CL_SUCCESS ) {
        return result;
    }
    (*output) = out_size;
    return result;
}

void* enqueue_map_buffer( cl_command_queue queue
                        , cl_mem buffer
                        , cl_bool blocking
                        , cl_map_flags map_flags
                        , size_t offset
                        , size_t size
                        , cl_uint num_events
                        , const cl_event* events
                        , cl_event* event
                        , cl_int* err_code )
{
    return clEnqueueMapBuffer( queue
                             , buffer
                             , blocking
                             , map_flags
                             , offset
                             , size
                             , num_events
                             , events
                             , event
                             , err_code );
}

cl_int enqueue_unmap_mem( cl_command_queue queue
                        , cl_mem mem
                        , void* ptr
                        , cl_uint num_events_in_wait_list
                        , const cl_event* events
                        , cl_event* ev )
{
    return clEnqueueUnmapMemObject( queue
                                  , mem
                                  , ptr
                                  , num_events_in_wait_list
                                  , events
                                  , ev );
}

cl_command_queue create_command_queue( cl_context context
                                     , cl_device_id device
                                     , cl_command_queue_properties props
                                     , cl_int* ret_code )
{
    return clCreateCommandQueue( context, device, props, ret_code );
}

cl_context create_context( const cl_context_properties* props
                         , cl_uint num_devices
                         , const cl_device_id* devices
                         , void CL_CALLBACK (*pfn_notify)
                           (const char*, const void*, size_t, void*)
                         , void* user_data
                         , cl_int* errcode_ret )
{
    return clCreateContext( props, num_devices, devices, pfn_notify, user_data, errcode_ret );
}

cl_mem create_buffer( cl_context ctx
                    , cl_mem_flags flags
                    , size_t size
                    , void* host_ptr
                    , cl_int* err_ptr )
{
    return clCreateBuffer( ctx, flags, size, host_ptr, err_ptr );
}

cl_int wait_for_events( cl_uint num_events, const cl_event* events )
{
    return clWaitForEvents( num_events, events );
}


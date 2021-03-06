-- | Raw enums and functions.
--
-- This is an internal module for this library.
--

{-# LANGUAGE CPP #-}

module Data.OpenCL.Raw where

import Data.Int
import Data.Word
import Foreign.C.Types
import Foreign.Ptr

#define CONSTANT_FUN(prefix) \
  foreign import ccall unsafe cl_##prefix :: Word32

CONSTANT_FUN(INVALID_LINKER_OPTIONS)
CONSTANT_FUN(LINK_PROGRAM_FAILURE)
CONSTANT_FUN(BUILD_PROGRAM_FAILURE)
CONSTANT_FUN(INVALID_COMPILER_OPTIONS)
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
CONSTANT_FUN(DEVICE_HALF_FP_CONFIG);

foreign import ccall unsafe code_success :: Int32
foreign import ccall unsafe code_platform_not_found :: Int32

foreign import ccall unsafe get_num_platforms :: Ptr Word32 -> IO Int32
foreign import ccall unsafe get_platforms :: Word32 -> Ptr CPlatformID -> IO Int32
foreign import ccall unsafe get_platform_info_size :: CPlatformID -> Word32 -> Ptr CSize -> IO Int32
foreign import ccall unsafe get_platform_info :: CPlatformID -> Word32 -> CSize -> Ptr CChar -> IO Int32
foreign import ccall get_num_devices :: CPlatformID -> Ptr Word32 -> IO Int32
foreign import ccall get_devices :: CPlatformID -> Word32 -> Ptr CDeviceID -> IO Int32
foreign import ccall get_device_info_size :: CDeviceID -> Word32 -> Ptr CSize -> IO Int32
foreign import ccall get_device_info :: CDeviceID -> Word32 -> CSize -> Ptr () -> IO Int32
foreign import ccall release_device :: CDeviceID -> IO ()
foreign import ccall release_context :: CContext -> IO ()
foreign import ccall retain_device :: CDeviceID -> IO ()
foreign import ccall retain_context :: CContext -> IO ()
foreign import ccall release_command_queue :: CCommandQueue -> IO ()
foreign import ccall release_event :: CEvent -> IO ()
foreign import ccall create_buffer :: CContext -> Word64 -> CSize -> Ptr () -> Ptr Int32 -> IO CMem
foreign import ccall release_mem :: CMem -> IO ()
foreign import ccall release_program :: CProgram -> IO ()
foreign import ccall release_kernel :: CKernel -> IO ()
foreign import ccall create_kernel :: CProgram -> Ptr CChar -> Ptr Int32 -> IO CKernel
foreign import ccall set_kernel_arg :: CKernel -> Word32 -> CSize -> Ptr () -> IO Int32
foreign import ccall wait_for_events :: Word32 -> Ptr CEvent -> IO Int32

foreign import ccall create_command_queue
  :: CContext
  -> CDeviceID
  -> Word64
  -> Ptr Int32
  -> IO CCommandQueue
foreign import ccall create_sub_devices
  :: CDeviceID
  -> Ptr CIntPtr
  -> Word32
  -> Ptr CDeviceID
  -> Ptr Word32
  -> IO Int32
foreign import ccall create_context
  :: Ptr CIntPtr
  -> Word32
  -> Ptr CDeviceID
  -> FunPtr CallbackFun
  -> Ptr ()
  -> Ptr Int32
  -> IO CContext
foreign import ccall enqueue_map_buffer
  :: CCommandQueue
  -> CMem
  -> Word32
  -> Word64
  -> CSize
  -> CSize
  -> Word32
  -> Ptr CEvent
  -> Ptr CEvent
  -> Ptr Int32
  -> IO (Ptr ())
foreign import ccall enqueue_unmap_mem
  :: CCommandQueue
  -> CMem
  -> Ptr ()
  -> Word32
  -> Ptr CEvent
  -> Ptr CEvent
  -> IO Int32
foreign import ccall create_program_with_source
  :: CContext
  -> Word32
  -> Ptr (Ptr CChar)
  -> Ptr CSize
  -> Ptr Int32
  -> IO CProgram
foreign import ccall get_program_build_info
  :: CProgram
  -> CDeviceID
  -> Word32
  -> CSize
  -> Ptr ()
  -> Ptr CSize
  -> IO Int32
foreign import ccall compile_program
  :: CProgram
  -> Word32
  -> Ptr CDeviceID
  -> Ptr CChar
  -> Word32
  -> Ptr CProgram
  -> Ptr (Ptr CChar)
  -> FunPtr (CProgram -> Ptr () -> IO ())
  -> Ptr ()
  -> IO Int32
foreign import ccall link_program
  :: CContext
  -> Word32
  -> Ptr CDeviceID
  -> Ptr CChar
  -> Word32
  -> Ptr CProgram
  -> FunPtr (CProgram -> Ptr () -> IO ())
  -> Ptr ()
  -> Ptr Int32
  -> IO CProgram
foreign import ccall enqueue_range_kernel
  :: CCommandQueue
  -> CKernel
  -> Word32
  -> Ptr CSize
  -> Ptr CSize
  -> Ptr CSize
  -> Word32
  -> Ptr CEvent
  -> Ptr CEvent
  -> IO Int32
foreign import ccall get_buffer_size
  :: CMem
  -> Ptr CSize
  -> IO Int32
foreign import ccall get_null_platform
  :: CPlatformID

type CallbackFun = Ptr CChar -> Ptr CChar -> CSize -> Ptr () -> IO ()

foreign import ccall "wrapper"
  wrapFun :: CallbackFun -> IO (FunPtr CallbackFun)

type CPlatformID   = Ptr ()
type CDeviceID     = Ptr ()
type CContext      = Ptr ()
type CCommandQueue = Ptr ()
type CMem          = Ptr ()
type CEvent        = Ptr ()
type CProgram      = Ptr ()
type CKernel       = Ptr ()


{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE CPP #-}

module Data.OpenCL
  ( CLFailure(..)
  , AffinityPartition(..)
  , Partitioning(..)
  , PartitionType(..)
  , CDeviceIDHandle()
  , CLCommandQueue()
  , CLEvent()
  , CLDevice(..)
  , CLDeviceCaps(..)
  , CLPlatform(..)
  , listPlatforms
  , listDevices
  , MemFlag(..)
  , MapFlag(..)
  , Block(..)
  , mapBuffer
  , unmapBuffer
  , withMappedBuffer
  , createSubDevices
  , createContext
  , createCommandQueue
  , createKernel
  , setKernelArgPtr
  , setKernelArgBuffer
  , setKernelArgStorable
  , createBufferRaw
  , createBufferUninitialized
  , createBufferFromBS
  , createBufferFromVector
  , createProgram
  , createProgramFromFilename
  , compileProgram
  , enqueueRangeKernel
  , linkProgram
  , waitEvents
  , CommandQueueProperty(..)
  , CLContext() )
  where

import Control.Concurrent
import Control.Monad
import Control.Monad.Catch
import Control.Monad.IO.Class
import Control.Monad.Primitive
import Data.Bits
import qualified Data.ByteString as B
import qualified Data.ByteString.Unsafe as B
import Data.Data
import Data.Foldable
import Data.Int
import Data.List ( groupBy )
import Data.Monoid
import Data.Traversable
import qualified Data.Vector.Storable as VS
import Data.Word
import Foreign hiding ( void )
import Foreign.C.String
import Foreign.C.Types
import GHC.Exts ( currentCallStack )
import GHC.Generics hiding ( L1 )

foreign import ccall unsafe code_success :: Int32

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

foreign import ccall get_num_platforms :: Ptr Word32 -> IO Int32
foreign import ccall get_platforms :: Word32 -> Ptr CPlatformID -> IO Int32
foreign import ccall get_platform_info_size :: CPlatformID -> Word32 -> Ptr CSize -> IO Int32
foreign import ccall get_platform_info :: CPlatformID -> Word32 -> CSize -> Ptr CChar -> IO Int32
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

data CLFailure
  = CLFailure Int32 [String]
  | CLCompilationFailure B.ByteString
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )

instance Exception CLFailure

clErrorify :: IO Int32 -> IO ()
clErrorify action = do
  result <- action
  if result == code_success
    then return ()
    else do stack <- currentCallStack
            throwM $ CLFailure result stack

data CLPlatform = CLPlatform
  { platformID         :: CPlatformID
  , platformProfile    :: String
  , platformVersion    :: String
  , platformName       :: String
  , platformVendor     :: String
  , platformExtensions :: [String] }
  deriving ( Eq, Ord, Show, Typeable, Data, Generic )

newtype CLProgram = CLProgram
  { handleProgram :: MVar CProgram }
  deriving ( Eq, Typeable, Generic )

newtype CLKernel = CLKernel
  { _handleKernel :: MVar CKernel }
  deriving ( Eq, Typeable, Generic )

newtype CLEvent = CLEvent
  { handleEvent :: MVar CEvent }
  deriving ( Eq, Typeable, Generic )

newtype CLCommandQueue = CLCommandQueue
  { _handleCommandQueue :: MVar CCommandQueue }
  deriving ( Eq, Typeable, Generic )

newtype CLMem = CLMem
  { _handleMem :: MVar CMem }
  deriving ( Eq, Typeable, Generic )

newtype CDeviceIDHandle = CDeviceIDHandle
  { handleDeviceID :: MVar CDeviceID }
  deriving ( Eq, Typeable, Generic )

makeUnmanagedCDeviceIDHandle :: CDeviceID -> IO CDeviceIDHandle
makeUnmanagedCDeviceIDHandle device_id =
  CDeviceIDHandle <$> newMVar device_id

makeManagedCDeviceIDHandle :: Maybe CDeviceID -> CDeviceID -> IO CDeviceIDHandle
makeManagedCDeviceIDHandle maybe_retain device_id = mask_ $ do
  mvar <- newMVar device_id
  let releasing = case maybe_retain of
                    Nothing -> return ()
                    Just r -> release_device r
  case maybe_retain of
    Nothing -> return ()
    Just r -> retain_device r

  void $ mkWeakMVar mvar $ do
    release_device device_id
    releasing

  return $ CDeviceIDHandle mvar

newtype CLContext = CLContext
  { _handleContext :: MVar CContext }
  deriving ( Eq, Typeable, Generic )

data CLDevice = CLDevice
  { deviceID :: CDeviceIDHandle
  , deviceParentDevice :: CDeviceID
  , devicePlatform :: CPlatformID
  , deviceCaps :: CLDeviceCaps }
  deriving ( Eq, Typeable, Generic )

data CLDeviceCaps = CLDeviceCaps
  { deviceAvailable                             :: Bool
  , deviceAddressBits                           :: Word64
  , deviceBuiltinKernels                        :: [String]
  , deviceCompilerAvailable                     :: Bool
  , deviceFpDenormSupported                     :: Bool
  , deviceFpInfNanSupported                     :: Bool
  , deviceFpRoundToNearestSupported             :: Bool
  , deviceFpRoundToZeroSupported                :: Bool
  , deviceFpRoundToInfSupported                 :: Bool
  , deviceFpFmaSupported                        :: Bool
  , deviceFpSoftFloat                           :: Bool
  , deviceEndianLittle                          :: Bool
  , deviceErrorCorrectionSupport                :: Bool
  , deviceExecKernelSupported                   :: Bool
  , deviceExecNativeKernelSupported             :: Bool
  , deviceExtensions                            :: [String]
  , deviceGlobalMemCacheSize                    :: Word64
  , deviceGlobalMemCacheType                    :: CacheType
  , deviceGlobalMemCachelineSize                :: Word64
  , deviceGlobalMemSize                         :: Word64
  , deviceHalfFpDenormSupported                 :: Bool
  , deviceHalfFpInfNanSupported                 :: Bool
  , deviceHalfFpRoundToNearestSupported         :: Bool
  , deviceHalfFpRoundToZeroSupported            :: Bool
  , deviceHalfFpRoundToInfSupported             :: Bool
  , deviceHalfFpFmaSupported                    :: Bool
  , deviceHalfFpSoftFloat                       :: Bool
  , deviceHostUnifiedMemory                     :: Bool
  , deviceImageSupport                          :: Bool
  , deviceImage2DMaxHeight                      :: CSize
  , deviceImage2DMaxWidth                       :: CSize
  , deviceImage3DMaxDepth                       :: CSize
  , deviceImage3DMaxHeight                      :: CSize
  , deviceImage3DMaxWidth                       :: CSize
  , deviceImageMaxBufferSize                    :: CSize
  , deviceImageMaxArraySize                     :: CSize
  , deviceLinkerAvailable                       :: Bool
  , deviceLocalMemSize                          :: Word64
  , deviceLocalMemType                          :: MemType
  , deviceMaxClockFrequency                     :: Word32
  , deviceMaxComputeUnits                       :: Word32
  , deviceMaxConstantArgs                       :: Word32
  , deviceMaxConstantBufferSize                 :: Word64
  , deviceMaxAllocSize                          :: Word64
  , deviceMaxParameterSize                      :: CSize
  , deviceMaxReadImageArgs                      :: Word64
  , deviceMaxSamplers                           :: Word64
  , deviceMaxWorkGroupSize                      :: CSize
  , deviceMaxWorkItemDimensions                 :: Word64
  , deviceMaxWorkItemSizes                      :: [CSize]
  , deviceMaxWriteImageArgs                     :: Word64
  , deviceMemBaseAddrAlign                      :: Word64
  , deviceMinDataTypeAlignSize                  :: Word64
  , deviceName                                  :: String
  , deviceNativeVectorWidthChar                 :: Word64
  , deviceNativeVectorWidthShort                :: Word64
  , deviceNativeVectorWidthInt                  :: Word64
  , deviceNativeVectorWidthLong                 :: Word64
  , deviceNativeVectorWidthFloat                :: Word64
  , deviceNativeVectorWidthDouble               :: Word64
  , deviceNativeVectorWidthHalf                 :: Word64
  , deviceOpenCLCVersion                        :: String
  , devicePartitionMaxSubDevices                :: Word64
  , devicePartitionEquallySupported             :: Bool
  , devicePartitionByCountsSupported            :: Bool
  , devicePartitionByAffinityDomainSupported    :: Bool
  , deviceAffinityDomainNumaSupported           :: Bool
  , deviceAffinityDomainL4Supported             :: Bool
  , deviceAffinityDomainL3Supported             :: Bool
  , deviceAffinityDomainL2Supported             :: Bool
  , deviceAffinityDomainL1Supported             :: Bool
  , deviceAffinityDomainNextPartitionable       :: Bool
  , devicePartitionType                         :: [PartitionType]
  , devicePrintfBufferSize                      :: CSize
  , devicePreferredInteropUserSync              :: Bool
  , devicePreferredVectorWidthChar              :: Word64
  , devicePreferredVectorWidthShort             :: Word64
  , devicePreferredVectorWidthInt               :: Word64
  , devicePreferredVectorWidthLong              :: Word64
  , devicePreferredVectorWidthFloat             :: Word64
  , devicePreferredVectorWidthDouble            :: Word64
  , devicePreferredVectorWidthHalf              :: Word64
  , deviceProfile                               :: String
  , deviceProfilingTimerResolution              :: CSize
  , deviceQueueProfilingSupported               :: Bool
  , deviceQueueOutOfOrderExecModeSupported      :: Bool
  , deviceSingleFpDenormSupported               :: Bool
  , deviceSingleFpInfNanSupported               :: Bool
  , deviceSingleFpRoundToNearestSupported       :: Bool
  , deviceSingleFpRoundToZeroSupported          :: Bool
  , deviceSingleFpRoundToInfSupported           :: Bool
  , deviceSingleFpFmaSupported                  :: Bool
  , deviceSingleFpCorrectlyRoundedDivideSqrt    :: Bool
  , deviceSingleFpSoftFloat                     :: Bool
  , deviceType                                  :: [DeviceType]
  , deviceVendor                                :: String
  , deviceVendorID                              :: Word64
  , deviceVersion                               :: String
  , deviceDriverVersion                         :: String }
  deriving ( Eq, Ord, Read, Show, Typeable, Generic )

data DeviceType
  = CPU
  | GPU
  | Accelerator
  | Default
  | Custom
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

data CacheType
  = NoCache
  | ReadCache
  | ReadWriteCache
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

data PartitionType
  = Equally
  | ByCounts
  | ByAffinityDomain
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

data MemType
  = Local
  | Global
  | NoMem
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

listDevices :: MonadIO m => CLPlatform -> m [CLDevice]
listDevices (CLPlatform { platformID = pid } ) = liftIO $ mask_ $ do
  num_devices <- alloca $ \ptr -> do
    clErrorify $ get_num_devices pid ptr
    peek ptr
  device_ids <- allocaArray (fromIntegral num_devices) $ \arr_ptr -> do
    clErrorify $ get_devices pid num_devices arr_ptr
    peekArray (fromIntegral num_devices) arr_ptr

  for device_ids $ \device_id -> do
    unmanaged_did <- makeUnmanagedCDeviceIDHandle device_id
    makeDevice unmanaged_did

hasBit :: Word64 -> Word32 -> Bool
hasBit thing bit = (thing .&. fromIntegral bit) /= 0

onCLFailure :: Int32 -> IO a -> a -> IO a
onCLFailure f action backup = do
  result <- try action
  case result of 
    Left (CLFailure code _) | code == f -> return backup
    Left exc -> throwM exc
    Right ok -> return ok

makeDevice :: CDeviceIDHandle -> IO CLDevice
makeDevice (CDeviceIDHandle did_handle) = do
  did <- readMVar did_handle
  let get_string_field fid = alloca $ \sz_ptr -> do
                         clErrorify $ get_device_info_size did
                                                           fid
                                                           sz_ptr
                         sz <- fromIntegral <$> peek sz_ptr
                         allocaBytes (sz+1) $ \cptr -> do
                           clErrorify $ get_device_info did
                                             fid
                                             (fromIntegral sz)
                                             cptr
                           pokeElemOff (castPtr cptr :: Ptr Word8) sz 0
                           peekCString (castPtr cptr)

      get_field :: Storable a => Word32 -> IO a
      get_field fid = alloca $ \v_ptr -> do
                        v <- peek v_ptr
                        clErrorify $ get_device_info
                                       did
                                       fid
                                       (fromIntegral $ sizeOf v)
                                       (castPtr v_ptr)
                        peek v_ptr
      {-# NOINLINE get_field #-}

      get_bool fid = do r <- get_field fid
                        return $ (r :: Word32) /= 0

  builtin_kernels <- get_string_field cl_DEVICE_BUILT_IN_KERNELS
  device_name <- get_string_field cl_DEVICE_NAME
  device_opencl_c_version <- get_string_field cl_DEVICE_OPENCL_C_VERSION
  device_profile <- get_string_field cl_DEVICE_PROFILE
  device_vendor <- get_string_field cl_DEVICE_VENDOR
  device_version <- get_string_field cl_DEVICE_VERSION
  driver_version <- get_string_field cl_DRIVER_VERSION
  exts <- get_string_field cl_DEVICE_EXTENSIONS

  available <- get_bool cl_DEVICE_AVAILABLE
  comp_available <- get_bool cl_DEVICE_COMPILER_AVAILABLE
  addr_bits <- get_field cl_DEVICE_ADDRESS_BITS

  fp_config <- get_field cl_DEVICE_DOUBLE_FP_CONFIG
  endian_little <- get_bool cl_DEVICE_ENDIAN_LITTLE
  error_correction <- get_bool cl_DEVICE_ERROR_CORRECTION_SUPPORT
  exec_capabilities <- get_field cl_DEVICE_EXECUTION_CAPABILITIES

  global_mem_cache_size <- get_field cl_DEVICE_GLOBAL_MEM_CACHE_SIZE
  global_mem_cache_type <- get_field cl_DEVICE_GLOBAL_MEM_CACHE_TYPE
  global_mem_cache_line_size <- get_field cl_DEVICE_GLOBAL_MEM_CACHELINE_SIZE
  global_mem_size <- get_field cl_DEVICE_GLOBAL_MEM_SIZE

  half_fp_config <- onCLFailure (-30)
                      (get_field cl_DEVICE_HALF_FP_CONFIG)
                      0
  host_unified_memory <- get_bool cl_DEVICE_HOST_UNIFIED_MEMORY
  image_support <- get_bool cl_DEVICE_IMAGE_SUPPORT
  max_height_2d <- get_field cl_DEVICE_IMAGE2D_MAX_HEIGHT
  max_width_2d <- get_field cl_DEVICE_IMAGE2D_MAX_WIDTH
  max_depth_3d <- get_field cl_DEVICE_IMAGE3D_MAX_DEPTH
  max_height_3d <- get_field cl_DEVICE_IMAGE3D_MAX_HEIGHT
  max_width_3d <- get_field cl_DEVICE_IMAGE3D_MAX_WIDTH
  linker_available <- get_bool cl_DEVICE_LINKER_AVAILABLE
  local_mem_size <- get_field cl_DEVICE_LOCAL_MEM_SIZE
  local_mem_type <- get_field cl_DEVICE_LOCAL_MEM_TYPE
  max_clock_freq <- get_field cl_DEVICE_MAX_CLOCK_FREQUENCY
  max_compute_units <- get_field cl_DEVICE_MAX_COMPUTE_UNITS
  max_constant_args <- get_field cl_DEVICE_MAX_CONSTANT_ARGS
  max_constant_buffer_size <- get_field cl_DEVICE_MAX_CONSTANT_BUFFER_SIZE
  max_mem_alloc_size <- get_field cl_DEVICE_MAX_MEM_ALLOC_SIZE
  max_parameter_size <- get_field cl_DEVICE_MAX_PARAMETER_SIZE
  max_read_image_args <- get_field cl_DEVICE_MAX_READ_IMAGE_ARGS
  max_samplers <- get_field cl_DEVICE_MAX_SAMPLERS
  max_work_group_size <- get_field cl_DEVICE_MAX_WORK_GROUP_SIZE
  max_work_item_dimensions <- get_field cl_DEVICE_MAX_WORK_ITEM_DIMENSIONS
  max_write_image_args <- get_field cl_DEVICE_MAX_WRITE_IMAGE_ARGS
  mem_base_addr_align <- get_field cl_DEVICE_MEM_BASE_ADDR_ALIGN
  min_data_type_align <- get_field cl_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE
  native_char <- get_field cl_DEVICE_NATIVE_VECTOR_WIDTH_CHAR
  native_short <- get_field cl_DEVICE_NATIVE_VECTOR_WIDTH_SHORT
  native_int <- get_field cl_DEVICE_NATIVE_VECTOR_WIDTH_INT
  native_long <- get_field cl_DEVICE_NATIVE_VECTOR_WIDTH_LONG
  native_float <- get_field cl_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT
  native_double <- get_field cl_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
  native_half <- get_field cl_DEVICE_NATIVE_VECTOR_WIDTH_HALF
  parent_device <- get_field cl_DEVICE_PARENT_DEVICE
  max_sub_devices <- get_field cl_DEVICE_PARTITION_MAX_SUB_DEVICES
  affinity_domain <- get_field cl_DEVICE_PARTITION_AFFINITY_DOMAIN
  platform <- get_field cl_DEVICE_PLATFORM
  preferred_char <- get_field cl_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR
  preferred_short <- get_field cl_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT
  preferred_int <- get_field cl_DEVICE_PREFERRED_VECTOR_WIDTH_INT
  preferred_long <- get_field cl_DEVICE_PREFERRED_VECTOR_WIDTH_LONG
  preferred_float <- get_field cl_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT
  preferred_double <- get_field cl_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE
  preferred_half <- get_field cl_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
  printf_buffer_size <- get_field cl_DEVICE_PRINTF_BUFFER_SIZE
  preferred_interop_user_sync <- get_bool cl_DEVICE_PREFERRED_INTEROP_USER_SYNC
  profiling_timer_resolution <- get_field cl_DEVICE_PROFILING_TIMER_RESOLUTION
  queue_properties <- get_field cl_DEVICE_QUEUE_PROPERTIES
  single_fp_config <- get_field cl_DEVICE_SINGLE_FP_CONFIG
  device_type <- get_field cl_DEVICE_TYPE
  device_vendor_id <- get_field cl_DEVICE_VENDOR_ID
  max_image_buffer_size <- get_field cl_DEVICE_IMAGE_MAX_BUFFER_SIZE
  max_image_array_size <- get_field cl_DEVICE_IMAGE_MAX_ARRAY_SIZE

  max_item_sizes <- allocaArray (fromIntegral max_work_item_dimensions) $ \(arr_ptr :: Ptr CSize) -> do
    clErrorify $ get_device_info did
                                 cl_DEVICE_MAX_WORK_ITEM_SIZES
                                 (fromIntegral $ sizeOf (undefined :: CSize) * fromIntegral max_work_item_dimensions)
                                 (castPtr arr_ptr)
    peekArray (fromIntegral max_work_item_dimensions) arr_ptr

  partitions <- alloca $ \sz_ptr -> do
    clErrorify $ get_device_info_size did
                                      cl_DEVICE_PARTITION_PROPERTIES
                                      sz_ptr
    sz <- peek sz_ptr
    arr <- allocaArray (fromIntegral sz) $ \(partpr :: Ptr CIntPtr) -> do
      clErrorify $ get_device_info did
                                   cl_DEVICE_PARTITION_PROPERTIES
                                   sz
                                   (castPtr partpr)
      peekArray (fromIntegral sz) partpr
    return $ fmap fromIntegral arr
  
  partition_type <- alloca $ \sz_ptr -> do
    clErrorify $ get_device_info_size did
                                      cl_DEVICE_PARTITION_TYPE
                                      sz_ptr
    sz <- peek sz_ptr
    arr <- allocaArray (fromIntegral sz) $ \(partpr :: Ptr CIntPtr) -> do
      clErrorify $ get_device_info did
                                   cl_DEVICE_PARTITION_TYPE
                                   sz
                                   (castPtr partpr)
      peekArray (fromIntegral sz) partpr
    return $ fmap fromIntegral arr

  touch did_handle

  return CLDevice {
      deviceID = CDeviceIDHandle did_handle
    , deviceParentDevice = parent_device
    , devicePlatform = platform
    , deviceCaps = CLDeviceCaps
    { deviceAvailable = available
    , devicePreferredInteropUserSync = preferred_interop_user_sync
    , deviceAddressBits = addr_bits
    , deviceBuiltinKernels = filter (/= ",") $
                             groupBy (\a b -> a /= ',' && b /= ',')
                             builtin_kernels
    , deviceCompilerAvailable = comp_available
    , deviceFpDenormSupported = hasBit fp_config cl_FP_DENORM
    , deviceFpInfNanSupported = hasBit fp_config cl_FP_INF_NAN
    , deviceFpRoundToNearestSupported = hasBit fp_config cl_FP_ROUND_TO_NEAREST
    , deviceFpRoundToZeroSupported = hasBit fp_config cl_FP_ROUND_TO_ZERO
    , deviceFpRoundToInfSupported = hasBit fp_config cl_FP_ROUND_TO_INF
    , deviceFpFmaSupported = hasBit fp_config cl_FP_FMA
    , deviceFpSoftFloat = hasBit fp_config cl_FP_SOFT_FLOAT
    , deviceEndianLittle = endian_little
    , deviceErrorCorrectionSupport = error_correction
    , deviceExecKernelSupported = hasBit exec_capabilities cl_EXEC_KERNEL
    , deviceExecNativeKernelSupported = hasBit exec_capabilities cl_EXEC_NATIVE_KERNEL
    , deviceExtensions = words exts
    , deviceGlobalMemCacheSize = global_mem_cache_size
    , deviceGlobalMemCacheType = if | global_mem_cache_type == cl_READ_ONLY_CACHE
                                      -> ReadCache
                                    | global_mem_cache_type == cl_READ_WRITE_CACHE
                                      -> ReadWriteCache
                                    | otherwise -> NoCache
    , deviceGlobalMemCachelineSize = global_mem_cache_line_size
    , deviceGlobalMemSize = global_mem_size
    , deviceHalfFpDenormSupported = hasBit half_fp_config cl_FP_DENORM
    , deviceHalfFpInfNanSupported = hasBit half_fp_config cl_FP_INF_NAN
    , deviceHalfFpRoundToNearestSupported = hasBit half_fp_config cl_FP_ROUND_TO_NEAREST
    , deviceHalfFpRoundToZeroSupported = hasBit half_fp_config cl_FP_ROUND_TO_ZERO
    , deviceHalfFpRoundToInfSupported = hasBit half_fp_config cl_FP_ROUND_TO_INF
    , deviceHalfFpFmaSupported = hasBit half_fp_config cl_FP_FMA
    , deviceHalfFpSoftFloat = hasBit half_fp_config cl_FP_SOFT_FLOAT
    , deviceHostUnifiedMemory = host_unified_memory
    , deviceImageSupport = image_support
    , deviceImage2DMaxHeight = max_height_2d
    , deviceImage2DMaxWidth = max_width_2d
    , deviceImage3DMaxDepth = max_depth_3d
    , deviceImage3DMaxHeight = max_height_3d
    , deviceImage3DMaxWidth = max_width_3d
    , deviceImageMaxBufferSize = max_image_buffer_size
    , deviceImageMaxArraySize = max_image_array_size
    , deviceLinkerAvailable = linker_available
    , deviceLocalMemSize = local_mem_size
    , deviceLocalMemType = toLocalMemType local_mem_type
    , deviceMaxClockFrequency = max_clock_freq
    , deviceMaxComputeUnits = max_compute_units
    , deviceMaxConstantArgs = max_constant_args
    , deviceMaxConstantBufferSize = max_constant_buffer_size
    , deviceMaxAllocSize = max_mem_alloc_size
    , deviceMaxParameterSize = max_parameter_size
    , deviceMaxReadImageArgs = max_read_image_args
    , deviceMaxSamplers = max_samplers
    , deviceMaxWorkGroupSize = max_work_group_size
    , deviceMaxWorkItemDimensions = max_work_item_dimensions
    , deviceMaxWorkItemSizes = max_item_sizes
    , deviceMaxWriteImageArgs = max_write_image_args
    , deviceMemBaseAddrAlign = mem_base_addr_align
    , deviceMinDataTypeAlignSize = min_data_type_align
    , deviceName = device_name
    , deviceNativeVectorWidthChar = native_char
    , deviceNativeVectorWidthShort = native_short
    , deviceNativeVectorWidthInt = native_int
    , deviceNativeVectorWidthLong = native_long
    , deviceNativeVectorWidthFloat = native_float
    , deviceNativeVectorWidthDouble = native_double
    , deviceNativeVectorWidthHalf = native_half
    , devicePreferredVectorWidthChar = preferred_char
    , devicePreferredVectorWidthShort = preferred_short
    , devicePreferredVectorWidthInt = preferred_int
    , devicePreferredVectorWidthLong = preferred_long
    , devicePreferredVectorWidthFloat = preferred_float
    , devicePreferredVectorWidthDouble = preferred_double
    , devicePreferredVectorWidthHalf = preferred_half
    , deviceOpenCLCVersion = device_opencl_c_version
    , devicePartitionMaxSubDevices = max_sub_devices
    , devicePartitionEquallySupported = elem cl_DEVICE_PARTITION_EQUALLY partitions
    , devicePartitionByCountsSupported = elem cl_DEVICE_PARTITION_BY_COUNTS partitions
    , devicePartitionByAffinityDomainSupported = elem cl_DEVICE_PARTITION_BY_AFFINITY_DOMAIN partitions
    , deviceAffinityDomainNumaSupported = flip hasBit cl_DEVICE_AFFINITY_DOMAIN_NUMA affinity_domain
    , deviceAffinityDomainL4Supported = flip hasBit cl_DEVICE_AFFINITY_DOMAIN_L4_CACHE affinity_domain
    , deviceAffinityDomainL3Supported = flip hasBit cl_DEVICE_AFFINITY_DOMAIN_L3_CACHE affinity_domain
    , deviceAffinityDomainL2Supported = flip hasBit cl_DEVICE_AFFINITY_DOMAIN_L2_CACHE affinity_domain
    , deviceAffinityDomainL1Supported = flip hasBit cl_DEVICE_AFFINITY_DOMAIN_L1_CACHE affinity_domain
    , deviceAffinityDomainNextPartitionable = flip hasBit cl_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE affinity_domain
    , devicePartitionType = concat $ fmap toPartitionType partition_type
    , devicePrintfBufferSize = printf_buffer_size
    , deviceProfile = device_profile
    , deviceProfilingTimerResolution = profiling_timer_resolution
    , deviceQueueProfilingSupported = flip hasBit cl_QUEUE_PROFILING_ENABLE queue_properties
    , deviceQueueOutOfOrderExecModeSupported = flip hasBit cl_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE queue_properties
    , deviceSingleFpDenormSupported = hasBit single_fp_config cl_FP_DENORM
    , deviceSingleFpInfNanSupported = hasBit single_fp_config cl_FP_INF_NAN
    , deviceSingleFpRoundToNearestSupported = hasBit single_fp_config cl_FP_ROUND_TO_NEAREST
    , deviceSingleFpRoundToZeroSupported = hasBit single_fp_config cl_FP_ROUND_TO_ZERO
    , deviceSingleFpRoundToInfSupported = hasBit single_fp_config cl_FP_ROUND_TO_INF
    , deviceSingleFpFmaSupported = hasBit single_fp_config cl_FP_FMA
    , deviceSingleFpCorrectlyRoundedDivideSqrt = hasBit single_fp_config cl_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT
    , deviceSingleFpSoftFloat = hasBit single_fp_config cl_FP_SOFT_FLOAT
    , deviceType = toDeviceTypes device_type
    , deviceVendor = device_vendor
    , deviceVendorID = device_vendor_id
    , deviceVersion = device_version
    , deviceDriverVersion = driver_version } }

toLocalMemType :: Word32 -> MemType
toLocalMemType x =
  if | x == cl_LOCAL -> Local
     | x == cl_GLOBAL -> Global
     | otherwise -> NoMem

toPartitionType :: CIntPtr -> [PartitionType]
toPartitionType (fromIntegral -> x) =
  if | x == cl_DEVICE_PARTITION_EQUALLY -> [Equally]
     | x == cl_DEVICE_PARTITION_BY_COUNTS -> [ByCounts]
     | x == cl_DEVICE_PARTITION_BY_AFFINITY_DOMAIN -> [ByAffinityDomain]
     | otherwise -> []

data AffinityPartition
  = Numa
  | L4
  | L3
  | L2
  | L1
  | NextPartitionable
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

toDeviceTypes :: Word64 -> [DeviceType]
toDeviceTypes x = concat [
   if hasBit x cl_DEVICE_TYPE_CPU
    then [CPU]
    else []
  ,if hasBit x cl_DEVICE_TYPE_GPU
    then [GPU]
    else []
  ,if hasBit x cl_DEVICE_TYPE_ACCELERATOR
    then [Accelerator]
    else []
  ,if hasBit x cl_DEVICE_TYPE_DEFAULT
    then [Default]
    else []
  ,if hasBit x cl_DEVICE_TYPE_CUSTOM
    then [Custom]
    else []]

data Partitioning
  = PartitionEqually !Int
  | PartitionByCounts [Int]
  | PartitionByAffinityDomain !AffinityPartition
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )

createSubDevices :: MonadIO m => CLDevice -> Partitioning -> m [CLDevice]
createSubDevices dev part = liftIO $ mask_ $ do
  did <- readMVar (handleDeviceID $ deviceID dev)
  flip finally (touch (handleDeviceID $ deviceID dev)) $
    (case part of
      PartitionEqually n ->
        withArray [fromIntegral cl_DEVICE_PARTITION_EQUALLY :: CIntPtr, fromIntegral n, 0]
      PartitionByCounts counts ->
        withArray ([fromIntegral cl_DEVICE_PARTITION_BY_COUNTS :: CIntPtr] <>
                   fmap fromIntegral counts <> [0])
      PartitionByAffinityDomain domain ->
        withArray ([fromIntegral cl_DEVICE_PARTITION_BY_AFFINITY_DOMAIN :: CIntPtr
                   ,fromIntegral $ case domain of
                     Numa -> cl_DEVICE_AFFINITY_DOMAIN_NUMA
                     L4 -> cl_DEVICE_AFFINITY_DOMAIN_L4_CACHE
                     L3 -> cl_DEVICE_AFFINITY_DOMAIN_L3_CACHE
                     L2 -> cl_DEVICE_AFFINITY_DOMAIN_L2_CACHE
                     L1 -> cl_DEVICE_AFFINITY_DOMAIN_L1_CACHE
                     NextPartitionable -> cl_DEVICE_AFFINITY_DOMAIN_NEXT_PARTITIONABLE
                   ,0])) $ \arr -> do
      alloca $ \sz_ptr -> do
        clErrorify $ create_sub_devices did
                                        arr
                                        0
                                        nullPtr
                                        sz_ptr
        sz <- peek sz_ptr
        dids <- allocaArray (fromIntegral sz) $ \device_arr -> do
          clErrorify $ create_sub_devices did
                                          arr
                                          sz
                                          device_arr
                                          nullPtr
          peekArray (fromIntegral sz) device_arr

        managed_dids <- for dids (makeManagedCDeviceIDHandle (Just did))
        for managed_dids makeDevice

listPlatforms :: MonadIO m => m [CLPlatform]
listPlatforms = liftIO $ do
  num_plats <- alloca $ \ptr -> do
    clErrorify $ get_num_platforms ptr
    peek ptr
  platform_ids <- allocaArray (fromIntegral num_plats) $ \arr_ptr -> do
    clErrorify $ get_platforms num_plats arr_ptr
    peekArray (fromIntegral num_plats) arr_ptr

  for platform_ids $ \platform_id -> do
    let get_field fid = alloca $ \sz_ptr -> do
                           clErrorify $ get_platform_info_size platform_id
                                                               fid
                                                               sz_ptr
                           sz <- fromIntegral <$> peek sz_ptr
                           allocaBytes (sz+1) $ \cptr -> do
                             clErrorify $ get_platform_info platform_id
                                               fid
                                               (fromIntegral sz)
                                               cptr
                             pokeElemOff cptr sz 0
                             peekCString cptr

    profile <- get_field cl_PLATFORM_PROFILE
    version <- get_field cl_PLATFORM_VERSION
    name <- get_field cl_PLATFORM_NAME
    vendor <- get_field cl_PLATFORM_VENDOR
    extensions <- get_field cl_PLATFORM_EXTENSIONS

    return CLPlatform
      { platformID = platform_id
      , platformProfile = profile
      , platformVersion = version
      , platformName = name
      , platformVendor = vendor
      , platformExtensions = words extensions }

createContext :: MonadIO m
              => (B.ByteString -> B.ByteString -> IO ())
              -> [CLDevice]
              -> m CLContext
createContext callback devices = liftIO $ mask_ $ do
  devs <- for devices $ \dev -> readMVar (handleDeviceID $ deviceID dev)
  flip finally (for_ devices $ \dev -> touch (handleDeviceID $ deviceID dev)) $
    withArray devs $ \devs_ptr -> do
      hs_fun <- wrapFun $ \errinfo private_info private_info_sz _user_data -> do
        bs <- B.packCString errinfo
        private_info <- B.packCStringLen (private_info, fromIntegral private_info_sz)
        callback bs private_info

      alloca $ \errcode_ptr -> do
        ctx <- create_context nullPtr
                              (fromIntegral $ length devices)
                              devs_ptr
                              hs_fun
                              nullPtr
                              errcode_ptr
        if ctx == nullPtr
          then do freeHaskellFunPtr hs_fun
                  errcode <- peek errcode_ptr
                  stack <- currentCallStack
                  throwM $ CLFailure errcode stack

          else do mvar <- newMVar ctx
                  for_ devs retain_device
                  void $ mkWeakMVar mvar $ do
                    freeHaskellFunPtr hs_fun
                    for_ devs release_device
                    release_context ctx
                  return $ CLContext mvar

data CommandQueueProperty
  = QueueOutOfOrder
  | EnableProfiling
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )

createCommandQueue :: MonadIO m
                   => CLContext
                   -> CLDevice
                   -> [CommandQueueProperty]
                   -> m CLCommandQueue
createCommandQueue (CLContext handle) device props = liftIO $ mask_ $ do
  ctx <- readMVar handle
  did <- readMVar $ handleDeviceID $ deviceID device
  flip finally (touch handle >> touch (handleDeviceID $ deviceID device)) $
    alloca $ \ret_ptr -> do
      queue <- create_command_queue ctx
                                    did
                                    flags
                                    ret_ptr
      ret <- peek ret_ptr
      clErrorify $ return ret

      retain_device did
      retain_context ctx

      mvar <- newMVar queue
      void $ mkWeakMVar mvar $ do
        release_command_queue queue
        release_device did
        release_context ctx

      return $ CLCommandQueue mvar
 where
  flags = fromIntegral $
          (if QueueOutOfOrder `elem` props
             then cl_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
             else 0) .|.
          (if EnableProfiling `elem` props
             then cl_QUEUE_PROFILING_ENABLE
             else 0)

data MemFlag
  = ReadWrite
  | ReadOnly
  | WriteOnly
  | UseHostPtr
  | AllocHostPtr
  | CopyHostPtr
  | HostWriteOnly
  | HostReadOnly
  | HostNoAccess
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

memFlagToBitfield :: MemFlag -> Word64
memFlagToBitfield mf = fromIntegral $ toField mf
 where
  toField ReadWrite     = cl_MEM_READ_WRITE
  toField ReadOnly      = cl_MEM_READ_ONLY
  toField WriteOnly     = cl_MEM_WRITE_ONLY
  toField UseHostPtr    = cl_MEM_USE_HOST_PTR
  toField AllocHostPtr  = cl_MEM_ALLOC_HOST_PTR
  toField CopyHostPtr   = cl_MEM_COPY_HOST_PTR
  toField HostWriteOnly = cl_MEM_HOST_WRITE_ONLY
  toField HostReadOnly  = cl_MEM_HOST_READ_ONLY
  toField HostNoAccess  = cl_MEM_HOST_NO_ACCESS

data Block = Block | NoBlock
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

doEnqueueing :: (Word32 -> Ptr CEvent -> Ptr CEvent -> IO Int32)
             -> [CLEvent]
             -> IO CLEvent
doEnqueueing action wait_events = do
  evs <- for wait_events $ readMVar . handleEvent
  flip finally (for_ wait_events $ touch . handleEvent) $
    withArray evs $ \evs_ptr ->
      alloca $ \result_evt -> mask_ $ do
        poke result_evt nullPtr
        clErrorify $ action (fromIntegral $ length wait_events)
                            (if null wait_events
                               then nullPtr
                               else evs_ptr)
                            result_evt
        ev <- peek result_evt
        ev_var <- newMVar ev
        void $ mkWeakMVar ev_var $ release_event ev
        return $ CLEvent ev_var

doEnqueueing2 :: (Word32 -> Ptr CEvent -> Ptr CEvent -> Ptr Int32 -> IO a)
              -> [CLEvent]
              -> IO (a, CLEvent)
doEnqueueing2 action wait_events = do
  evs <- for wait_events $ readMVar . handleEvent
  flip finally (for_ wait_events $ touch . handleEvent) $
    withArray evs $ \evs_ptr ->
      alloca $ \result_evt ->
        alloca $ \ret_val -> do
          poke result_evt nullPtr
          result <- action (fromIntegral $ length wait_events)
                           (if null wait_events
                              then nullPtr
                              else evs_ptr)
                           result_evt
                           ret_val
          ret <- peek ret_val
          clErrorify $ return ret

          ev <- peek result_evt
          ev_var <- newMVar ev
          void $ mkWeakMVar ev_var $ release_event ev
          return $ (result, CLEvent ev_var)

data MapFlag
  = MapRead
  | MapWrite
  | MapWriteInvalidate
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

mapFlagToBitfield :: MapFlag -> Word64
mapFlagToBitfield MapRead = fromIntegral cl_MAP_READ
mapFlagToBitfield MapWrite = fromIntegral cl_MAP_WRITE
mapFlagToBitfield MapWriteInvalidate = fromIntegral cl_MAP_WRITE_INVALIDATE_REGION

mapBuffer :: MonadIO m
          => CLCommandQueue
          -> CLMem
          -> Block
          -> [MapFlag]
          -> Int
          -> Int
          -> [CLEvent]
          -> m (Ptr a, CLEvent)
mapBuffer (CLCommandQueue queue_mvar) (CLMem mem_mvar) block map_flags offset sz wait_events = liftIO $ do
  queue <- readMVar queue_mvar
  mem <- readMVar mem_mvar
  (ptr, ev) <- flip finally (touch queue_mvar >> touch mem_mvar) $
    doEnqueueing2
      (enqueue_map_buffer queue
                          mem
                          (case block of
                            Block -> 1
                            NoBlock -> 0)
                          map_bitfield
                          (fromIntegral offset)
                          (fromIntegral sz))
      wait_events
  return (castPtr ptr, ev)
 where
  map_bitfield = foldr (.|.) 0 $ fmap mapFlagToBitfield map_flags

unmapBuffer :: MonadIO m
            => CLCommandQueue
            -> CLMem
            -> Ptr ()
            -> [CLEvent]
            -> m CLEvent
unmapBuffer (CLCommandQueue queue_mvar) (CLMem mem_mvar) ptr events = liftIO $ do
  queue <- readMVar queue_mvar
  mem <- readMVar mem_mvar
  flip finally (touch queue_mvar >> touch mem_mvar) $
    doEnqueueing (enqueue_unmap_mem queue mem ptr) events

withMappedBuffer :: (MonadIO m, MonadMask m)
                 => CLCommandQueue
                 -> CLMem
                 -> Block
                 -> [MapFlag]
                 -> Int
                 -> Int
                 -> [CLEvent]
                 -> (Ptr a -> CLEvent -> m b)
                 -> m b
withMappedBuffer queue mem block map_flags offset sz events action = do
  (ptr, ev) <- liftIO $ mapBuffer queue mem block map_flags offset sz events
  finally (action (castPtr ptr) ev) (unmapBuffer queue mem ptr [])

createBufferUninitialized :: MonadIO m
                          => CLContext
                          -> [MemFlag]
                          -> Int
                          -> m CLMem
createBufferUninitialized ctx flags sz = createBufferRaw ctx flags sz nullPtr

createBufferRaw :: MonadIO m
                => CLContext
                -> [MemFlag]
                -> Int
                -> Ptr ()
                -> m CLMem
createBufferRaw (CLContext ctx_mvar) memflags sz ptr = liftIO $ mask_ $ do
  ctx <- readMVar ctx_mvar
  flip finally (touch ctx_mvar) $ alloca $ \err_ptr -> do
    mem <- create_buffer ctx
                         flags
                         (fromIntegral sz)
                         ptr
                         err_ptr
    err <- peek err_ptr
    clErrorify $ return err
    retain_context ctx

    mvar <- newMVar mem
    void $ mkWeakMVar mvar $ do
      release_context ctx
      release_mem mem

    return $ CLMem mvar
 where
  flags = foldr (.|.) 0 (fmap memFlagToBitfield memflags)

createBufferFromBS :: MonadIO m
                   => CLContext
                   -> [MemFlag]
                   -> B.ByteString
                   -> m CLMem
createBufferFromBS ctx memflags bs = liftIO $ do
  B.unsafeUseAsCStringLen bs $ \(cstr, len) ->
    createBufferRaw ctx memflags len (castPtr cstr)

createBufferFromVector :: forall m s.
                          (MonadIO m
                          ,Storable s)
                       => CLContext
                       -> [MemFlag]
                       -> VS.Vector s
                       -> m CLMem
createBufferFromVector ctx memflags vec = liftIO $ do
  VS.unsafeWith vec $ \vec_ptr ->
    createBufferRaw ctx
                    memflags
                    (sizeOf (undefined :: s) * VS.length vec)
                    (castPtr vec_ptr)

createProgramFromFilename :: MonadIO m
                          => CLContext
                          -> FilePath
                          -> m CLProgram
createProgramFromFilename ctx source_filename = liftIO $ do
  bs <- B.readFile source_filename
  createProgram ctx bs

createProgram :: MonadIO m
              => CLContext
              -> B.ByteString
              -> m CLProgram
createProgram (CLContext ctx_var) source = liftIO $ mask_ $ do
  ctx <- readMVar ctx_var
  flip finally (touch ctx_var) $ do
    B.unsafeUseAsCStringLen source $ \(cstr, len) ->
      with cstr $ \cstr_ptr ->
      with (fromIntegral len) $ \len_ptr ->
      alloca $ \err_ptr -> do
        program <- create_program_with_source ctx
                                              1
                                              cstr_ptr
                                              len_ptr
                                              err_ptr
        err <- peek err_ptr
        clErrorify $ return err

        var <- newMVar program
        void $ mkWeakMVar var $ release_program program
        return $ CLProgram var

getProgramLog :: CProgram
              -> CDeviceID
              -> IO B.ByteString
getProgramLog program dev =
  alloca $ \sz_ptr -> do
    clErrorify $ get_program_build_info program
                                        dev
                                        cl_PROGRAM_BUILD_LOG
                                        0
                                        nullPtr
                                        sz_ptr
    sz <- peek sz_ptr
    allocaBytes (fromIntegral $ sz+1) $ \cstr -> do
      clErrorify $ get_program_build_info program
                                          dev
                                          cl_PROGRAM_BUILD_LOG
                                          sz
                                          (castPtr cstr)
                                          nullPtr
      pokeElemOff cstr (fromIntegral sz) 0
      B.packCString cstr

compileProgram :: MonadIO m
               => CLProgram
               -> [CLDevice]
               -> B.ByteString
               -> m B.ByteString
compileProgram _ [] _ = error "compileProgram: must use at least once device."
compileProgram (CLProgram program_var) devices options = liftIO $ mask_ $ do
  program <- readMVar program_var
  devs <- for devices $ readMVar . handleDeviceID . deviceID
  flip finally (touch program_var >> for_ devices (\d -> touch $ handleDeviceID $ deviceID d)) $
    withArray devs $ \devs_ptr ->
      B.useAsCString options $ \options_ptr -> do
        result <- compile_program program
                                  (fromIntegral $ length devices)
                                  devs_ptr
                                  options_ptr
                                  0
                                  nullPtr
                                  nullPtr
                                  (castPtrToFunPtr nullPtr)
                                  nullPtr
        log <- getProgramLog program (head devs)
        () <- case result of
          code | code == fromIntegral cl_INVALID_COMPILER_OPTIONS ->
            throwM $ CLCompilationFailure log
          code | code == fromIntegral cl_COMPILE_PROGRAM_FAILURE ||
                 code == fromIntegral cl_BUILD_PROGRAM_FAILURE ->
            throwM $ CLCompilationFailure log
          code | code /= code_success -> do
            stack <- currentCallStack
            throwM $ CLFailure code stack
          _ -> return ()
        return log

linkProgram :: MonadIO m
            => CLContext
            -> [CLDevice]
            -> B.ByteString
            -> [CLProgram]
            -> m (CLProgram, B.ByteString)
linkProgram _ [] _ _ = error "linkProgram: must use at least one device."
linkProgram _ _ _ [] = error "linkProgram: must use at least one program."
linkProgram (CLContext ctx_var) devices options programs = liftIO $ mask_ $ do
  ctx <- readMVar ctx_var
  devs <- for devices $ readMVar . handleDeviceID . deviceID
  progs <- for programs $ readMVar . handleProgram
  flip finally (do touch ctx_var
                   for_ devices (touch . handleDeviceID . deviceID)
                   for_ programs (touch . handleProgram)) $
    withArray devs $ \devs_ptr ->
    withArray progs $ \progs_ptr ->
    B.useAsCString options $ \options_ptr ->
    alloca $ \errcode_ptr -> do
      result <- link_program ctx
                             (fromIntegral $ length devices)
                             devs_ptr
                             options_ptr
                             (fromIntegral $ length programs)
                             progs_ptr
                             (castPtrToFunPtr nullPtr)
                             nullPtr
                             errcode_ptr
      prog_var <- newMVar result
      unless (result == nullPtr) $
        void $ mkWeakMVar prog_var $ release_program result
      errcode <- peek errcode_ptr
      log <- getProgramLog (head progs) (head devs)
      if | errcode == fromIntegral cl_INVALID_LINKER_OPTIONS ||
           errcode == fromIntegral cl_LINK_PROGRAM_FAILURE
           -> throwM $ CLCompilationFailure log
         | errcode /= code_success
           -> do stack <- currentCallStack
                 throwM $ CLFailure errcode stack
         | otherwise -> return ()
      return (CLProgram prog_var, log)

createKernel :: MonadIO m
             => CLProgram
             -> B.ByteString
             -> m CLKernel
createKernel (CLProgram prog_var) kernel_name = liftIO $ mask_ $ do
  prog <- readMVar prog_var
  flip finally (touch prog_var) $ B.useAsCString kernel_name $ \kernel_name_ptr ->
    alloca $ \err_ptr -> do
      kernel <- create_kernel prog kernel_name_ptr err_ptr
      err <- peek err_ptr
      clErrorify $ return err

      kernel_var <- newMVar kernel
      void $ mkWeakMVar kernel_var $ release_kernel kernel

      return $ CLKernel kernel_var

setKernelArgPtr :: MonadIO m
                => CLKernel
                -> Int
                -> Int
                -> Ptr ()
                -> m ()
setKernelArgPtr (CLKernel kernel_var) arg_index arg_size arg_ptr = liftIO $ mask_ $ do
  kernel <- readMVar kernel_var
  flip finally (touch kernel_var) $
    clErrorify $ set_kernel_arg kernel
                                (fromIntegral arg_index)
                                (fromIntegral arg_size)
                                arg_ptr
{-# INLINE setKernelArgPtr #-}

setKernelArgBuffer :: MonadIO m
                   => CLKernel
                   -> Int
                   -> CLMem
                   -> m ()
setKernelArgBuffer (CLKernel kernel_var) arg_index (CLMem mem_var) = liftIO $ mask_ $ do
  mem <- readMVar mem_var
  kernel <- readMVar kernel_var
  flip finally (touch mem_var >> touch kernel_var) $ do
    with mem $ \mem_ptr ->
      clErrorify $ set_kernel_arg kernel
                                  (fromIntegral arg_index)
                                  (fromIntegral $ sizeOf (undefined :: CMem))
                                  (castPtr mem_ptr)
{-# INLINE setKernelArgBuffer #-}

setKernelArgStorable :: forall s m. (Storable s, MonadIO m)
                     => CLKernel
                     -> Int
                     -> s
                     -> m ()
setKernelArgStorable kernel arg_index storable = liftIO $
  with storable $ \storable_ptr ->
    setKernelArgPtr kernel arg_index (sizeOf (undefined :: s)) (castPtr storable_ptr)
{-# INLINE setKernelArgStorable #-}

enqueueRangeKernel :: MonadIO m
                   => CLCommandQueue
                   -> CLKernel
                   -> [Int]
                   -> [Int]
                   -> [Int]
                   -> [CLEvent]
                   -> m CLEvent
enqueueRangeKernel (CLCommandQueue command_var) (CLKernel kernel_var)
                   offset work_size workgroup_size
                   wait_events
  | length offset /= length work_size ||
    length offset /= length workgroup_size ||
    length work_size /= length workgroup_size
      = error "enqueueRangeKernel: dimensions of offset, work size and workgroup size must be the same."
  | length offset < 1 || length offset > 3
      = error "enqueueRangeKernel: dimensions must be between 1 and 3."
  | otherwise = liftIO $ mask_ $ do
    command <- readMVar command_var
    kernel <- readMVar kernel_var
    flip finally (do touch command_var
                     touch kernel_var) $
      withArray (fmap fromIntegral offset) $ \offset_arr ->
      withArray (fmap fromIntegral work_size) $ \work_arr ->
      withArray (fmap fromIntegral workgroup_size) $ \workgroup_arr ->
        doEnqueueing
          (enqueue_range_kernel command
                                kernel
                                (fromIntegral $ length offset)
                                offset_arr
                                work_arr
                                workgroup_arr)
          wait_events

waitEvents :: MonadIO m => [CLEvent] -> m ()
waitEvents wait_events = liftIO $ do
  evs <- for wait_events $ readMVar . handleEvent
  flip finally (for_ wait_events $ touch . handleEvent) $
    withArray evs $ \evs_array ->
      clErrorify $ wait_for_events (fromIntegral $ length wait_events)
                                   evs_array


{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE MultiWayIf #-}
{-# LANGUAGE CPP #-}

module Data.OpenCL
  ( OpenCL()
  , makeOpenCL
  , CLPlatform(..)
  , CLDevice(..)
  , CLDeviceCaps(..)
  , listPlatforms
  , listDevices )
  where

import Control.Concurrent
import Control.Exception
import Control.Monad
import Control.Monad.IO.Class
import Control.Monad.Primitive
import Data.Aeson
import Data.Bits
import Data.Data
import Data.Int
import Data.IORef
import Data.List ( groupBy )
import Data.Monoid
import Data.Traversable
import Data.Word
import Foreign hiding ( void )
import Foreign.C.String
import Foreign.C.Types
import GHC.Exts ( currentCallStack )
import GHC.Generics
import System.IO.Unsafe

foreign import ccall unsafe code_success :: Int32

#define CONSTANT_FUN(prefix) \
  foreign import ccall unsafe cl_##prefix :: Word32

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

type CPlatformID = Ptr ()
type CDeviceID   = Ptr ()

indexSupply :: IORef Integer
indexSupply = unsafePerformIO $ newIORef 0
{-# NOINLINE indexSupply #-}

data CLFailure = CLFailure Int32 [String]
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )

instance Exception CLFailure

data OpenCL = OpenCL
  { workMVar :: !(MVar (IO ()))
  , index    :: !Integer }
  deriving ( Typeable, Generic )

instance Eq OpenCL where
  (OpenCL _ i1) == (OpenCL _ i2) = i1 == i2

instance Ord OpenCL where
  (OpenCL _ i1) `compare` (OpenCL _ i2) = i1 `compare` i2

instance Show OpenCL where
  show (OpenCL _ i) = "OpenCL<" <> show i <> ">"

data SilentDeath = SilentDeath
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )

instance Exception SilentDeath

openclQueueThread :: MVar (IO ()) -> IO ()
openclQueueThread mvar = do
  result <- try $ forever $ do
    join $ takeMVar mvar
  let _ = result :: Either SilentDeath ()
  return ()

makeOpenCL :: MonadIO m => m OpenCL
makeOpenCL = liftIO $ mask_ $ do
  mvar <- newEmptyMVar
  tid <- forkIOWithUnmask $ \unmask -> unmask $
    if rtsSupportsBoundThreads
      then runInBoundThread $ openclQueueThread mvar
      else openclQueueThread mvar

  void $ mkWeakMVar mvar $ throwTo tid SilentDeath

  idx <- atomicModifyIORef' indexSupply $ \old -> 
           ( old+1, old )

  return OpenCL { workMVar = mvar
                , index = idx }

safeCall :: MonadIO m => OpenCL -> IO a -> m a
safeCall (OpenCL mvar _) action = liftIO $ do
  result <- newEmptyMVar
  putMVar mvar $ do
    r <- try $ action
    putMVar result r
  r <- takeMVar result
  touch mvar
  case r of
    Left (exc :: SomeException) -> throwIO exc
    Right ok -> return ok

callSafe :: MonadIO m => IO a -> OpenCL -> m a
callSafe = flip safeCall

clErrorify :: IO Int32 -> IO ()
clErrorify action = do
  result <- action
  if result == code_success
    then return ()
    else do stack <- currentCallStack
            throwIO $ CLFailure result stack

data CLPlatform = CLPlatform
  { platformID         :: CPlatformID
  , platformProfile    :: String
  , platformVersion    :: String
  , platformName       :: String
  , platformVendor     :: String
  , platformExtensions :: [String] }
  deriving ( Eq, Ord, Show, Typeable, Data, Generic )

data CLDevice = CLDevice
  { deviceID :: CDeviceID
  , deviceParentDevice :: CDeviceID
  , devicePlatform :: CPlatformID
  , deviceCaps :: CLDeviceCaps }
  deriving ( Eq, Ord, Show, Typeable, Generic )

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

instance FromJSON CSize where
  parseJSON thing = do
    x <- parseJSON thing
    return $ fromIntegral (x :: Word64)

instance ToJSON CSize where
  toJSON x = toJSON $ (fromIntegral x :: Word64)

instance FromJSON CLDeviceCaps
instance ToJSON CLDeviceCaps

data DeviceType
  = CPU
  | GPU
  | Accelerator
  | Default
  | Custom
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

instance FromJSON DeviceType
instance ToJSON DeviceType

data CacheType
  = NoCache
  | ReadCache
  | ReadWriteCache
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

instance FromJSON CacheType
instance ToJSON CacheType

data PartitionType
  = Equally
  | ByCounts
  | ByAffinityDomain
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

instance FromJSON PartitionType
instance ToJSON PartitionType

data MemType
  = Local
  | Global
  | NoMem
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

instance FromJSON MemType
instance ToJSON MemType

listDevices :: MonadIO m => OpenCL -> CLPlatform -> m [CLDevice]
listDevices opencl (CLPlatform { platformID = pid } ) = safeCall opencl $ do
  num_devices <- alloca $ \ptr -> do
    clErrorify $ get_num_devices pid ptr
    peek ptr
  device_ids <- allocaArray (fromIntegral num_devices) $ \arr_ptr -> do
    clErrorify $ get_devices pid num_devices arr_ptr
    peekArray (fromIntegral num_devices) arr_ptr

  for device_ids $ \device_id ->
    makeDevice device_id

hasBit :: Word64 -> Word32 -> Bool
hasBit thing bit = (thing .&. fromIntegral bit) /= 0

onCLFailure :: Int32 -> IO a -> a -> IO a
onCLFailure f action backup = do
  result <- try action
  case result of 
    Left (CLFailure code _) | code == f -> return backup
    Left exc -> throwIO exc
    Right ok -> return ok

makeDevice :: CDeviceID -> IO CLDevice
makeDevice did = do
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

  return CLDevice {
      deviceID = did
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


listPlatforms :: MonadIO m => OpenCL -> m [CLPlatform]
listPlatforms = callSafe $ do
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


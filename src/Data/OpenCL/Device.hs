-- | OpenCL devices.
--

{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE MultiWayIf #-}

module Data.OpenCL.Device
  ( CLDevice()
  , CLDeviceCaps(..)
  , listDevices
  , createSubDevices
  , DeviceType(..)
  , MemType(..)
  , PartitionType(..)
  , deviceCaps )
  where

import Control.Concurrent.MVar
import Control.Monad
import Control.Monad.Catch
import Control.Monad.IO.Class
import Control.Monad.Primitive
import Data.Bits
import Data.Data
import Data.List ( groupBy )
import Data.Monoid
import Data.OpenCL.Exception
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import Data.Traversable
import Data.Word
import Foreign.C.String
import Foreign.C.Types
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable
import GHC.Generics hiding ( L1 )

-- | Given a platform, return devices it supports.
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

hasBit :: Word64 -> Word32 -> Bool
hasBit thing bit = (thing .&. fromIntegral bit) /= 0

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

-- | Partitions a device into subdevices.
--
-- Check the `CLDeviceCaps` structure for supported partitionings for a
-- particular device.
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

-- | Affinities to partition with.
data AffinityPartition
  = Numa
  | L4
  | L3
  | L2
  | L1
  | NextPartitionable
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

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

-- | Describes a partitioning scheme.
data Partitioning
  = PartitionEqually !Int
  | PartitionByCounts [Int]
  | PartitionByAffinityDomain !AffinityPartition
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )


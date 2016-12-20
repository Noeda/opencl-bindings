-- | Internal module that exposes the innards of managed OpenCL handles.
--

{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}

module Data.OpenCL.Handle
  ( CLCommandQueue(..)
  , CLContext(..)
  , CLDevice(..)
  , CLDeviceCaps(..)
  , CLEvent(..)
  , CLKernel(..)
  , CLMem(..)
  , CLPlatform(..)
  , CLProgram(..)
  , CDeviceIDHandle(..)
  , DeviceType(..)
  , PartitionType(..)
  , MemType(..)
  , CacheType(..) )
  where

import Control.Concurrent.MVar
import Data.Data
import Data.OpenCL.Raw
import Data.Yaml
import Data.Word
import Foreign.C.Types
import GHC.Generics

instance ToJSON CSize where
  toJSON sz = toJSON (fromIntegral sz :: Word64)

instance FromJSON CSize where
  parseJSON ob = (fromIntegral :: Word64 -> CSize) <$> parseJSON ob

-- | Managed handle to OpenCL device.
--
-- Devices are garbage collected automatically.
data CLDevice = CLDevice
  { deviceID :: CDeviceIDHandle
  , deviceParentDevice :: CDeviceID
  , devicePlatform :: CPlatformID
  , deviceCaps :: CLDeviceCaps }
  deriving ( Eq, Typeable, Generic )

newtype CDeviceIDHandle = CDeviceIDHandle
  { handleDeviceID :: MVar CDeviceID }
  deriving ( Eq, Typeable, Generic )

-- | Managed handle to OpenCL platform handle.
data CLPlatform = CLPlatform
  { platformID         :: CPlatformID
  , platformProfile    :: String
  , platformVersion    :: String
  , platformName       :: String
  , platformVendor     :: String
  , platformExtensions :: [String] }
  deriving ( Eq, Ord, Show, Typeable, Data, Generic )

-- | Managed handle to OpenCL memory, which may or may not be on OpenCL device.
newtype CLMem = CLMem
  { _handleMem :: MVar CMem }
  deriving ( Eq, Typeable, Generic )

-- | What kind of device is this exactly.
data DeviceType
  = CPU
  | GPU
  | Accelerator
  | Default
  | Custom
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum, ToJSON, FromJSON )

-- | Supported partitioning types for the device.
data PartitionType
  = Equally
  | ByCounts
  | ByAffinityDomain
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum, ToJSON, FromJSON )

-- | What kind of memory is this.
data MemType
  = Local
  | Global
  | NoMem
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum, ToJSON, FromJSON )

-- | Cache type.
data CacheType
  = NoCache
  | ReadCache
  | ReadWriteCache
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum, ToJSON, FromJSON )

-- | Device caps. This record can be used to inspect hardware capabilities of
-- an OpenCL device (use `deviceCaps` to get it out of `CLDevice`).
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
  deriving ( Eq, Ord, Read, Show, Typeable, Generic, ToJSON, FromJSON )

-- | Managed handle to OpenCL context.
newtype CLContext = CLContext
  { _handleContext :: MVar CContext }
  deriving ( Eq, Typeable, Generic )

-- | Managed handle to OpenCL event. You can wait on `CLEvent` to complete.
newtype CLEvent = CLEvent
  { handleEvent :: MVar CEvent }
  deriving ( Eq, Typeable, Generic )

-- | Managed handle to OpenCL command queue.
newtype CLCommandQueue = CLCommandQueue
  { _handleCommandQueue :: MVar CCommandQueue }
  deriving ( Eq, Typeable, Generic )

newtype CLKernel = CLKernel
  { _handleKernel :: MVar CKernel }
  deriving ( Eq, Typeable, Generic )

newtype CLProgram = CLProgram
  { handleProgram :: MVar CProgram }
  deriving ( Eq, Typeable, Generic )


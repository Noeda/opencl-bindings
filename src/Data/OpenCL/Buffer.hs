-- | OpenCL buffers, dealing with memory on device.
--

{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

module Data.OpenCL.Buffer
  ( CLMem()
  , MemFlag(..)
  , Block(..)
  , mapBuffer
  , unmapBuffer
  , withMappedBuffer
  , createBufferRaw
  , createBufferUninitialized
  , createBufferFromBS
  , createBufferFromVector )
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
import Data.OpenCL.Event.Internal
import Data.OpenCL.Exception
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import qualified Data.Vector.Storable as VS
import Data.Word
import Foreign.Marshal.Alloc
import Foreign.Ptr
import Foreign.Storable
import GHC.Generics

-- | Memory usage flag.
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

-- | Do you want this call to block or not?
data Block = Block | NoBlock
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic, Enum )

-- | Describes how to map.
data MapFlag
  = MapRead
  | MapWrite
  | MapWriteInvalidate  -- ^ Map for writing but invalidate existing buffer for the mapped chunk.
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

-- | Maps a buffer to host memory.
--
-- If you use `NoBlock` then the pointer may not be immediately valid. Use the
-- provided `CLEvent` to wait before using the pointer.
mapBuffer :: MonadIO m
          => CLCommandQueue
          -> CLMem
          -> Block
          -> [MapFlag]
          -> Int           -- ^ Offset from which to map.
          -> Int           -- ^ How many bytes to map.
          -> [CLEvent]     -- ^ Wait on these events before continuing.
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

-- | Unmaps a previously mapped buffer.
unmapBuffer :: MonadIO m
            => CLCommandQueue
            -> CLMem
            -> Ptr ()
            -> [CLEvent]     -- ^ Wait on these events before continuing.
            -> m CLEvent
unmapBuffer (CLCommandQueue queue_mvar) (CLMem mem_mvar) ptr events = liftIO $ do
  queue <- readMVar queue_mvar
  mem <- readMVar mem_mvar
  flip finally (touch queue_mvar >> touch mem_mvar) $
    doEnqueueing (enqueue_unmap_mem queue mem ptr) events

-- | Run an action with mapped buffer.
withMappedBuffer :: (MonadIO m, MonadMask m)
                 => CLCommandQueue
                 -> CLMem
                 -> Block
                 -> [MapFlag]
                 -> Int
                 -> Int
                 -> [CLEvent]
                 -> (Ptr a -> CLEvent -> m b)  -- ^ Invoked with pointer and a `CLEvent`. You can use `CLEvent` to wait until the pointer is ready (if you used `NoBlock`).
                 -> m (CLEvent, b) -- ^ The `CLEvent` can be waited on to wait until unmap completes.
withMappedBuffer queue mem block map_flags offset sz events action = do
  (ptr, ev) <- liftIO $ mapBuffer queue mem block map_flags offset sz events
  mvar <- liftIO $ newEmptyMVar
  result <- finally (action (castPtr ptr) ev) (liftIO . putMVar mvar =<< unmapBuffer queue mem ptr [])
  wev <- liftIO $ takeMVar mvar
  return (wev, result)

createBufferUninitialized :: MonadIO m
                          => CLContext
                          -> [MemFlag]
                          -> Int
                          -> m CLMem
createBufferUninitialized ctx flags sz = createBufferRaw ctx flags sz nullPtr

-- | Creates a buffer from a pointer and size.
--
-- If your pointer is not a null pointer, then you need to use one of the host
-- flags from `MemFlag`s.
createBufferRaw :: MonadIO m
                => CLContext
                -> [MemFlag]
                -> Int       -- ^ Number of bytes.
                -> Ptr ()    -- ^ Pointer to data.
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

-- | Creates a buffer from a strict bytestring. No copy of the bytestring is
-- made as far as Haskell goes (OpenCL might still do it).
--
-- You need to use one of the host `MemFlag`s with this.
createBufferFromBS :: MonadIO m
                   => CLContext
                   -> [MemFlag]
                   -> B.ByteString
                   -> m CLMem
createBufferFromBS ctx memflags bs = liftIO $ do
  B.unsafeUseAsCStringLen bs $ \(cstr, len) ->
    createBufferRaw ctx memflags len (castPtr cstr)

-- | Creates a buffer from a storable vector. No copy of the vector is made as
-- far as Haskell goes (OpenCL might still do it).
--
-- You need to use one of the host `MemFlag`s with this.
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

mapFlagToBitfield :: MapFlag -> Word64
mapFlagToBitfield MapRead = fromIntegral cl_MAP_READ
mapFlagToBitfield MapWrite = fromIntegral cl_MAP_WRITE
mapFlagToBitfield MapWriteInvalidate = fromIntegral cl_MAP_WRITE_INVALIDATE_REGION


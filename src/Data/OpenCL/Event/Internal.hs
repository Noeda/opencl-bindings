module Data.OpenCL.Event.Internal
  ( doEnqueueing
  , doEnqueueing2 )
  where

import Control.Concurrent.MVar
import Control.Monad
import Control.Monad.Catch
import Control.Monad.Primitive
import Foreign.Marshal.Alloc
import Foreign.Marshal.Array
import Foreign.Ptr
import Foreign.Storable
import Data.Foldable
import Data.Int
import Data.OpenCL.Exception
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import Data.Traversable
import Data.Word

-- | Utility function for a common raw OpenCL call.
--
-- Many OpenCL functions take a list of event pointers that you can pass to
-- wait for those events before enqueueing the command. This function takes
-- care of passing managed `CLEvent`s through this common pattern.
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

-- | Same as `doEnqueueing` but for a slightly different call pattern.
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

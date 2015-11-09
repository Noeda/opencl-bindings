-- | Event module. Wait until OpenCL commands are done.

module Data.OpenCL.Event
  ( CLEvent()
  , waitEvents )
  where

import Control.Concurrent.MVar
import Control.Monad.IO.Class
import Control.Monad.Catch
import Control.Monad.Primitive
import Data.Foldable
import Data.OpenCL.Exception
import Data.OpenCL.Handle
import Data.OpenCL.Raw
import Data.Traversable
import Foreign.Marshal.Array

-- | Waits until all events listed are done.
waitEvents :: MonadIO m => [CLEvent] -> m ()
waitEvents wait_events = liftIO $ do
  evs <- for wait_events $ readMVar . handleEvent
  flip finally (for_ wait_events $ touch . handleEvent) $
    withArray evs $ \evs_array ->
      clErrorify $ wait_for_events (fromIntegral $ length wait_events)
                                   evs_array


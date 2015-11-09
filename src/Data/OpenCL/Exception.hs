-- | Exceptions thrown by these OpenCL bindings.
--

{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}

module Data.OpenCL.Exception
  ( CLFailure(..)
  , clErrorify
  , onCLFailure )
  where

import Control.Monad.Catch
import qualified Data.ByteString as B
import Data.Data
import Data.Int
import Data.OpenCL.Raw
import GHC.Exts ( currentCallStack )
import GHC.Generics

-- | Thrown when some OpenCL function throws an error.
--

data CLFailure
  = CLFailure Int32 [String]
  | CLCompilationFailure B.ByteString
  deriving ( Eq, Ord, Show, Read, Typeable, Data, Generic )

instance Exception CLFailure

-- | Given a function returning an `Int32`, runs it and checks the return value
-- and throws a `CLFailure` if the value indicates OpenCL failure.
--
-- Used internally by this library to make OpenCL calls throw exceptions on
-- error.
clErrorify :: IO Int32 -> IO ()
clErrorify action = do
  result <- action
  if result == code_success
    then return ()
    else do stack <- currentCallStack
            throwM $ CLFailure result stack

-- | If given action throws a `CLFailure` with the given error code, then
-- return default value.
--
-- Used internally.
--
-- @
--   x <- onCLFailure 12345 (throwIO $ CLFailure 12345 []) 'a'
--   print x       -- 'a'
-- @
onCLFailure :: Int32 -> IO a -> a -> IO a
onCLFailure f action backup = do
  result <- try action
  case result of 
    Left (CLFailure code _) | code == f -> return backup
    Left exc -> throwM exc
    Right ok -> return ok


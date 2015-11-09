{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveDataTypeable #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE MultiWayIf #-}

module Data.OpenCL
  ( module Data.OpenCL.Buffer
  , module Data.OpenCL.CommandQueue
  , module Data.OpenCL.Context
  , module Data.OpenCL.Device
  , module Data.OpenCL.Event
  , module Data.OpenCL.Kernel
  , module Data.OpenCL.Platform
  , module Data.OpenCL.Program
  , CLFailure(..) )
  where

import Data.OpenCL.Buffer
import Data.OpenCL.CommandQueue
import Data.OpenCL.Context
import Data.OpenCL.Device
import Data.OpenCL.Event
import Data.OpenCL.Exception
import Data.OpenCL.Kernel
import Data.OpenCL.Platform
import Data.OpenCL.Program


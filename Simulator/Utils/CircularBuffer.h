/**
 * @file CircularBuffer.h
 * @author Jardi A. M. Jordan (jardiamj@gmail.com)
 * @date 01-22-2023
 * Supervised by Dr. Michael Stiber, UW Bothell CSSE Division
 * @ingroup Simulator/Utils
 * 
 * @brief A basic Template Circular Buffer 
 * 
 * This Circular Buffer is intended for implementing the Event Queuing Model of
 * the NG-911 system. The capacity of the buffer can be defined on instantiation
 * and you are only allowed to resize an empty buffer.
 * 
 * Trying to put an element into a buffer that is full results in an error,
 * captured via an assertion. Similarly, if the buffer is empty the value returned
 * by get() is, naturally, invalid. The onus is on the programmer to check if the
 * buffer is empty or full before retrieving or inserting values.
 * 
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <optional>
#include <vector>

template <typename T> class CircularBuffer {
public:
   /// @brief  Create a circular buffer with the given capacity
   ///
   /// Create and size the buffer so it can hold a set number of elements. If the
   /// size is set on instantiation, the buffer should not be resize.
   /// The size of the underlying data structure is set to capacity+1, to distinguish
   /// between a full and empty buffer.
   ///
   /// @param capacity  The number of elements that the buffer can hold, defaults to zero
   CircularBuffer(int capacity = 0) : buffer_(capacity + 1)
   {
      clear();
   }

   /// @brief  Resize the circular buffer
   /// @pre    Current buffer must be empty (Without valid elements)
   /// @param capacity  The number of elements that the buffer can hold
   void resize(int capacity)
   {
      // We are only allowed to resize an empty buffer
      assert(isEmpty());
      buffer_.resize(capacity + 1);
      // Ensure that the buffer is cleared
      clear();
   }

   /// @brief  Inserts the given element at the front of the queue
   /// @pre    The buffer must not be full
   /// @param element   The element to be inserted into the queue
   void put(T element)
   {
      // We throw an error if the buffer is full
      assert(!isFull());

      // Insert the new element and increment the front index
      buffer_[front_] = element;
      front_ = (front_ + 1) % buffer_.size();
   }

   /// @brief  Retrieves the element at the end of the queue
   ///
   /// The method returns the element wrapped in an std::optional data type. If the
   /// buffer is empty we return an empty std::optional constructed from std::nullopt.
   ///
   /// @post   The buffer will have one less element if it was not empty.
   /// @return The element at the end of the buffer, or std::nullopt if it is empty.
   std::optional<T> get()
   {
      if (isEmpty()) {
         // Return an empty optional object
         return std::nullopt;
      }

      // Get the value at the end of the queue and free up a space
      T value = buffer_[end_];
      end_ = (end_ + 1) % buffer_.size();

      return value;
   }

   /// @brief  Retrieves the element at the end of the buffer withouth dequeueing it
   ///
   /// @post   The elements in the buffer remain the same
   /// @return The element at the end of the buffer, or std::nulloopt if it is empty.
   std::optional<T> peek() const
   {
      std::optional<T> value;
      if (!isEmpty()) {
         value = buffer_[end_];
      }
      return value;
   }

   /// @brief Clears the circular buffer
   void clear()
   {
      front_ = 0;
      end_ = 0;
   }

   /// @brief Returns `true` if the buffer is empty, `false` otherwise
   /// @return `true` if the buffer is empty, `false` otherwise
   bool isEmpty() const
   {
      return front_ == end_;
   }

   /// @brief Returns `true` if the buffer is full, `false` otherwise
   ///
   /// We distinguish between an empty and full buffer by always having an unused space
   /// between the front and the end of the queue. A full buffer has the front one
   /// space behind the end of the queue.
   ///
   /// @return `true` if the buffer is full, `false` otherwise
   bool isFull() const
   {
      return ((front_ + 1) % buffer_.size()) == end_;
   }

   /// @brief Retrieves the number of elements that this buffer can hold
   ///
   /// Note that the underlying vector is sized one over the buffer capacity to
   /// distinguish between a full and empty buffer.
   ///
   /// @return The number of element that this buffer can hold
   size_t capacity() const
   {
      return buffer_.size() - 1;
   }

   /// @brief Retrieves the number of elements currently stored in the buffer
   /// @return The number of elements currently stored in the buffer
   size_t size() const
   {
      if (front_ >= end_) {
         return front_ - end_;
      }

      // if end_ is greater than front we substract the spaces between them
      // from the buffer's capacity
      return buffer_.size() + front_ - end_;
   }

private:
   /// Container for holding the buffer elements
   std::vector<T> buffer_;
   /// Index of the element at the start of the circular queue
   size_t front_ = 0;
   /// Index of the location where the next element will be stored in the circular queue
   size_t end_ = 0;
};
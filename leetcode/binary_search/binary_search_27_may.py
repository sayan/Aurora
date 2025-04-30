# 704. Binary Search
# Easy
# Topics
# Companies
# Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

# You must write an algorithm with O(log n) runtime complexity.


# Example 1:

# Input: nums = [-1,0,3,5,9,12], target = 9
# Output: 4
# Explanation: 9 exists in nums and its index is 4
# Example 2:

# Input: nums = [-1,0,3,5,9,12], target = 2
# Output: -1
# Explanation: 2 does not exist in nums so return -1


# Constraints:

# 1 <= nums.length <= 104
# -104 < nums[i], target < 104
# All the integers in nums are unique.
# nums is sorted in ascending order.

from typing import List


class Solutions:
    def search(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                ## target is smaller than mid, present in left half
                r = mid - 1
            else:
                ## target is greater than mid, present in right half
                l = mid + 1
        return -1


Solutions().search(nums=[-1, 0, 3, 5, 9, 12], target=9)
Solutions().search(nums=[-1, 0, 3, 5, 9, 12], target=2)


#################################################################
#################################################################
# 35. Search Insert Position
# Easy
# Topics
# Companies
# Given a sorted array of distinct integers and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

# You must write an algorithm with O(log n) runtime complexity.


# Example 1:

# Input: nums = [1,3,5,6], target = 5
# Output: 2
# Example 2:

# Input: nums = [1,3,5,6], target = 2
# Output: 1
# Example 3:

# Input: nums = [1,3,5,6], target = 7
# Output: 4


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        l = 0
        r = len(nums) - 1
        while l <= r:
            mid = (l + r) // 2
            # print(f"l: {l}, mid: {mid}, r: {r}")
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                ##  target is smaller will be on left side
                r = mid - 1
            else:
                l = mid + 1
        return mid + 1 if target > nums[mid] else min(mid - 1, 0)


Solution().insertPos(nums=[1, 3, 5, 6], target=0)

* 多重背包问题配合二进制优化
![问题描述](picture/multi-bag.png)
求解过程中注意到，si的值较大，此时可以用二进制优化的方式来进行解决，本质上就是类似将一个数用二进制进行表示（一开始原始方法是类似1+1+1+...的方式），现在也就是转化为1+2+4+...+rest=si的形式。对于小于si的每一个数都可以用1，2，4，...，rest来表示，且每个数能保证至多使用一次，因此二进制优化是合理的。
**困扰点：对于si-rest到si之间的数如何表示产生疑问，后来想到能用1，2，4，...表示si-2*rest到si-rest之间的数，再加上rest就能表示si-rest到si之间的数**
最终的时间复杂度为o(NV*logs)
最终解决的代码如下所示：
```c++
#include<iostream>
#include<vector>
int main()
{
    int num;
    int total_weight;
    std::cin>>num>>total_weight;
    std::vector<int> weight(2000000,0);
    std::vector<int> value(2000000,0);
    int tmp1;
    int tmp2;
    int tmp3;
    int count1=0;
    int count2=0;
    int k;
    for(int i=0;i<num;i++)
    {
        std::cin>>tmp1>>tmp2>>tmp3;
        for(k=1;k<=tmp3;k*=2)
        {
            weight[count1++]=k*tmp1;
            value[count2++]=k*tmp2;
            tmp3-=k;
        }
        if(tmp3>0)
        {
            weight[count1++]=tmp3*tmp1;
            value[count2++]=tmp3*tmp2;
        }
    }
    std::vector<int> dp(total_weight+1,0);
    for(int i=0;i<count1;i++)
    {
        for(int j=total_weight;j>=weight[i];j--)
        {
            dp[j]=std::max(dp[j],dp[j-weight[i]]+value[i]);
        }
    }
    std::cout<<dp[total_weight]<<std::endl;
    return 0;
}
```
---
* 单调栈
![单调栈](picture/sorted-stack.png)
暴力方法需要O(n^2)的时间复杂度，运用单调栈的思想，这里将构造一个栈，从栈顶到栈底单调递减
```c++
#include<iostream>
const int N = 1e5 + 10;
int stk[N],tt;
int main()
{
    int n;
    std::cin>>n;
    int tmp;
    for(int i=0;i<n;i++)
    {
        std::cin>>tmp;
        while(tt>0&&stk[tt]>=tmp) tt--;
        if(tt) printf("%d ",stk[tt]);
        else printf("-1 ");
        stk[++tt]=tmp;
    }
    return 0;
}
```
---
* 滑动窗口
![问题描述1](picture/slide-window1.png)
![问题描述2](picture/slide-window2.png)
该问题运用单调栈的思想，又因为有窗口大小的限制，因此想到运用单调队列，**队尾运用单调栈的思想，队头要保证在窗口内。**
```c++
#include<iostream>
const int N = 1e6 + 10;
int q[N],a[N],hh,tt=-1;
int main()
{
    int n,k;
    std::cin>>n>>k;
    int tmp;
    for(int i=0;i<n;i++)
    {
        std::cin>>tmp;
        a[i]=tmp;
    }
    for(int i=0;i<n;i++)
    {
        if((i-k+1)>q[hh]) hh++;
        while(tt>=hh&&a[q[tt]]>=a[i]) tt--;
        q[++tt]=i;
        if(i>=(k-1)) printf("%d ",a[q[hh]]);
    }
    printf("\n");
    tt=-1;
    hh=0;
    for(int i=0;i<n;i++)
    {
        if((i-k+1)>q[hh]) hh++;
        while(tt>=hh&&a[q[tt]]<=a[i]) tt--;
        q[++tt]=i;
        if(i>=(k-1)) printf("%d ",a[q[hh]]);
    }
    return 0;
}
```
* 多重背包问题配合滑动窗口优化
![问题描述](picture/multi-bag-slide_window.png)
相比于用二进制优化的方法，本题中物体的数量和体积数量级又再次增长,这会造成二进制优化的算法在大数量级上的时间复杂度依然较高，需要进一步优化，采用滑动窗口的方法进行解决。
滑动窗口的基本思想见上面的单调队列，大致的推导如下：
f(i,v)=max(f(i-1,v),f(i-1,v-w[i])+x[i],f(i-1,v-2* w[i])+2* x[i],...,f(i-1,v-s* w[i])+s*x[i])  <-- **将这个推导过程与滑动窗口联系起来，因为每个物品的数量是一定的**
f(i,v-w[i])=        max(f(i-1,v-w[i]),f(i-1,v-2*w[i])+x[i], ... )
详细的推导可以看 https://www.acwing.com/solution/content/53507/
得到最终的代码如下：
```c++
#include<iostream>
#include<cstring>
const int V=20010;
int x[V];//current
int y[V];//former
const int N=1010;
int v[N],w[N],s[N];
int tt,hh;
int q[V];
int main()
{
    int num;
    int total_weight;
    std::cin>>num>>total_weight;
    int tmp;
    for(int i=0;i<num;i++)
    {
        std::cin>>tmp;
        v[i]=tmp;
        std::cin>>tmp;
        w[i]=tmp;
        std::cin>>tmp;
        s[i]=tmp;
    }
    for(int i=0;i<num;i++)
    {
        memcpy(y,x,V*4);//注意这里memcpy中第三个参数是字节！！！！
        for(int r=0;r<v[i];r++)
        {
            tt=-1;
            hh=0;
            for(int j=r;j<=total_weight;j+=v[i])
            {
                if(tt>=hh&&(q[hh]<(j-v[i]*s[i]))) hh++;
                while(tt>=hh&&(y[q[tt]]+(j-q[tt])/v[i]*w[i])<=y[j]) tt--;
                q[++tt]=j;
                x[j]=y[q[hh]]+(j-q[hh])/v[i]*w[i];
            }
        }
    }
    printf("%d\n",x[total_weight]);
    return 0;
}
```
---



* 都是正整数时的给定目标和的长度最小的子数组

​	给定一个含有 `n` 个正整数的数组和一个正整数 `target` **。**

​	找出该数组中满足其总和大于等于 `target` 的长度最小的 **连续子数组** `[numsl, numsl+1, ..., numsr-1, numsr]` ，并返回其长度**。**如果不存在符合条件的子数组，返回 `0` 。



题解：运用双指针的思想，给定slow，fast不断向前移动，直到当前区间的和大于等于target，此时前移slow，直到当前区间的和小于target，这一步相当于是在放缩满足条件的区间，放缩完毕后，此时slow到达了第一次不满足的点，再固定slow，重复上面的过程，直到fast移出了数组，步骤结束。



相关函数如下：

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int tmp = 0;
        int res = 1e5+10;
        int slow = 0;
        for(int fast = 0;fast<nums.size();fast++)
        {
            tmp+=nums[fast];
            if(tmp>=target)
            {
                while(slow<=fast)
                {
                    tmp-=nums[slow];
                    if(tmp<target)
                    {
                        break;
                    }
                    slow++;
                }
                res=min(res,fast-slow+1);
                slow++;
            }
        }
        if (res<=1e5)
            return res;
        else
            return 0;
    }
};
```

* 环形链表

给定一个链表的头节点  `head` ，返回链表开始入环的第一个节点。 *如果链表无环，则返回 `null`。*

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 `pos` 来表示链表尾连接到链表中的位置（**索引从 0 开始**）。如果 `pos` 是 `-1`，则在该链表中没有环。**注意：`pos` 不作为参数进行传递**，仅仅是为了标识链表的实际情况。

**不允许修改** 链表。

**想法**：一个快指针，一个慢指针，这个在之前的练习中可以想得到，但是下一步是从起点出发，从相遇点出发并再次相遇，后面的这个思路很重要。

即：

 //m+n=x

 //m+n+k*cycle=2*x

 //m+n=k*cycle

 //m=k*cycle-n=(k>=1)(k-1)cycle+(cycle-n)

代码如下：

```c++
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
 //m+n=x
 //m+n+k*cycle=2*x
 //m+n=k*cycle
 //m=k*cycle-n=(k>=1)(k-1)cycle+(cycle-n)
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if(head==NULL) return NULL;
        ListNode* fast=head;
        ListNode* slow=head;
        do {
            fast=fast->next;
            if(!fast) return NULL;
            fast=fast->next;
            if(!fast) return NULL;
            slow=slow->next;
        } while(fast!=slow);
        slow=head;
        while(slow!=fast)
        {
            slow=slow->next;
            fast=fast->next;
        }
        return slow;
    }
};
```
* 四数之和
给你四个整数数组 nums1、nums2、nums3 和 nums4 ，数组长度都是 n ，请你计算有多少个元组 (i, j, k, l) 能满足：

0 <= i, j, k, l < n
nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0

**想法**：要把时间复杂度尽量控制到n^2，而不要让时间复杂度到n^3，代码如下：
```c++
#include<unordered_map>
class Solution {
public:
    int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4) {
        unordered_map<int,int> map;
       for(auto num1:nums1)
       {
           for(auto num2:nums2)
           {
               map[num1+num2]++;
           }
       } 
       int res=0;
       for(auto num3:nums3)
       {
           for(auto num4:nums4)
           {
               if(map.count(-num3-num4))
               {
                   res+=map[-num3-num4];
               }
           }
       }
       return res;
    }
};
```
* 同一个数组中找到三数之和为0
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，同时还满足 nums[i] + nums[j] + nums[k] == 0 。请

你返回所有和为 0 且不重复的三元组。

注意：答案中不可以包含重复的三元组。

**方法1**：运用哈希表的方法，将原本暴力的三维转化为二维，但是这个过程中要对于a，b，c分别去重
```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size();i++)
        {
            if(nums[i]>0) break;
            if(i>0&&nums[i]==nums[i-1])
                continue;
            unordered_set<int> set;
            for(int j=(i+1);j<nums.size();j++)
            {
                if(j>(i+2)&&nums[j]==nums[j-1]&&nums[j-1]==nums[j-2])//对于b去重
                    continue;
                int c = -nums[i]-nums[j];
                if(set.find(c)!=set.end())
                {
                    ans.push_back({nums[i],nums[j],c});
                    set.erase(c);//当有连续两个数相等时，对于c进行去重
                }
                else
                    set.insert(nums[j]);
            }
        }
        return ans;
    }
};
```
**方法2**：双指针
```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        sort(nums.begin(),nums.end());
        for(int i=0;i<nums.size();i++)
        {
            if(nums[i]>0) break;
            if(i>0&&nums[i]==nums[i-1])
                continue;
            int l = i+1;
            int r = nums.size()-1;
            while(l<r)
            {
                if(nums[i]+nums[l]+nums[r]>0)
                {
                    r--;
                }
                else if(nums[i]+nums[l]+nums[r]<0)
                {
                    l++;
                }
                else
                {
                    ans.push_back({nums[i],nums[l],nums[r]});
                    while(r>l&&nums[r]==nums[r-1]) r--;// 去重
                    while(r>l&&nums[l]==nums[l+1]) l++;// 去重
                    l++;
                    r--;
                }
                
            }
        }
        return ans;
    }
};
```

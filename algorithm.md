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



* 单词的翻转

给你一个字符串 `s` ，请你反转字符串中 **单词** 的顺序。

**单词** 是由非空格字符组成的字符串。`s` 中使用至少一个空格将字符串中的 **单词** 分隔开。

返回 **单词** 顺序颠倒且 **单词** 之间用单个空格连接的结果字符串。

**注意：**输入字符串 `s`中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中，单词间应当仅用单个空格分隔，且不包含任何额外的空格。



**难点** ：难点在于O(1)的空间复杂度，考虑先将多余的空格给删除，调用erase不是很理智，时间复杂度可能会到达O(n^2)，采用双指针的方式进行翻转，slow相当于遍历现在的字符串，fast相当于遍历原先的字符串。而后翻转可以看作先翻转整个字符串，再翻转之中的每个单词。

得到代码如下：

```c++
class Solution {
public:
    void removeWhiteSpace(string& s)
    {
        int fast=0;
        int slow=0;
        while(fast<s.length()&&s[fast]==' ')
        {
            fast++;
        }
        for(fast=fast;fast<s.length();fast++)
        {
            if(fast>0&&s[fast]==s[fast-1]&&s[fast]==' ')
                continue;
            else
                s[slow++]=s[fast];
        }
        slow--;
        for(slow;slow>0;slow--)
        {
            if(s[slow]!=' ')
                break;
        }
        s.resize(slow+1);
    }
    void reversestring(string& s,int start,int end)
    {
        for(int i=start;i<=(start+end)/2;i++)
        {
            swap(s[i],s[start+end-i]);
        }
    }
    string reverseWords(string s) {
        removeWhiteSpace(s);
        reversestring(s,0,s.length()-1);
        int former=0;
        int latter=0;
        int index;
        while(former<s.length())
        {
            for(index=former;index<s.length();index++)
            {
                if((s[index]==' ')||(index==(s.length()-1)))
                    break;
            }
            if(s[index]==' ')
                index--;
            latter=index;
            reversestring(s,former,latter);
            former=latter+2;
        }
        return s;
    }   
};
```

* 重复的子字符串

给定一个非空的字符串 `s` ，检查是否可以通过由它的一个子串重复多次构成。

**想法**：1. 运用KMP的思想。   2. 运用移动匹配的思想，去头去尾后查找原字符串，若能找到，则有重复子串。证明方式如下：

t=s+s，设去头去尾后s在t中的起始位置为i，有i大于0且i小于n，有s[0:n-1]=t[i:n+i-1]，以前一个s和后一个s的边界进行划分，则有s[0:n-i-1]=t[i:n-1]且s[n-i:n-1]=t[n:n+i-1]=t[0:i-1]，将t对应回s，则有s[0:n-i-1]=s[i:n-1]，s[n-i:n-1]=s[0:i-1]，相当于将s的前i个字符保持不变，移动到s的末尾，得到的新字符串与s相同，即在模n的意义下，有s[j]=s[j+i]，这对于任意的j都成立。不断地连写这个等式则有：

**s[j]=s[j+i]=s[j+2i]=...**

那么所有满足j0=j+k*i的位置 j0都有 s[j]=s[j0]，j 和 j0在模 i的意义下等价。由于我们已经在模 n 的意义下讨论这个问题，因此 j 和 j0在模 gcd(n,i) 的意义下等价，其中 gcd表示最大公约数。也就是说，字符串 sss 中的两个位置如果在模 gcd(n,i) 的意义下等价，那么它们对应的字符必然是相同的。

由于 gcd(n,i)一定是 n 的约数，那么字符串 s一定可以由其长度为 gcd(n,i)的前缀重复 n/gcd(n,i)次构成。

**方法1**

```c++
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int j=-1;
        vector<int> next(s.length(),-1);
        for(int i=1;i<s.length();i++)
        {
            while(j>=0&&s[i]!=s[j+1])
            {
                j=next[j];
            }
            if(s[i]==s[j+1])
                j++;
            next[i]=j;
        }
        if(next[s.length()-1]==-1) return false;
        if((s.length()%(s.length()-next[s.length()-1]-1))==0) return true;
        else return false;
    }
};
```

**方法2**

```c++
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int j=-1;
        vector<int> next(s.length(),-1);
        for(int i=1;i<s.length();i++)
        {
            while(j>=0&&s[i]!=s[j+1])
            {
                j=next[j];
            }
            if(s[i]==s[j+1])
                j++;
            next[i]=j;
        }
        if(next[s.length()-1]==-1) return false;
        if((s.length()%(s.length()-next[s.length()-1]-1))==0) return true;
        else return false;
    }
};
```

* 滑动窗口最大值

给你一个整数数组 `nums`，有一个大小为 `k` 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 `k` 个数字。滑动窗口每次只向右移动一位。

返回 *滑动窗口中的最大值* 。

**思路**：运用单调队列的思想来做，最大值在头部，但是要保证头部所对应的index满足滑动窗口大小的条件，尾部用于添加新数据，但是这里如果原来的尾部index对应的值已经小于等于将要加入队列的值，此时应该将尾部往左移，因为原本的这个值永远都不会进行输出，在这两个过程中还要考虑队列tt>=hh的条件。

则代码实现如下：

```c++
const int N = 1e5+10;
int qu[N];
int tt,hh;
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        tt=-1;
        hh=0;
        if(nums.size()<k)
        {
            vector<int> v;
            return v;
        }
        vector<int> ans(nums.size()-k+1,0);
        for(int i=0;i<nums.size();i++)
        {
            while(((hh<=tt)&&((i-qu[hh]+1)>k)))
            {
                ++hh;
            }
            while(((hh<=tt)&&(nums[qu[tt]]<=nums[i])))
                tt--;
            qu[++tt]=i;
            if(i>=(k-1))
            {
                ans[i-k+1]=nums[qu[hh]];
            }
        }
        return ans;
    }
};
```

* 前k个高频的元素

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回其中出现频率前 `k` 高的元素。你可以按 **任意顺序** 返回答案。

 **思路**：维护一个大小为k的小顶堆，当堆的大小大于k时，将堆顶的元素出堆，最后倒叙把堆中的元素输出（因为堆顶的为较小频率的值）。看到要对于频率进行排序，想到运用大顶堆或者小顶堆。

代码如下：

```c++
class Solution {
public:
    class mycomparison {
    public:
        bool operator()(const pair<int,int>& lp,const pair<int,int>& rp)
        {
            return lp.second>rp.second;
        }
    };
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int,int> map;
        for(auto num:nums)
        {
            map[num]++;
        }
        priority_queue<pair<int,int>,vector<pair<int,int>>,mycomparison> pq;
        for(auto e:map)
        {
            pq.push(e);
            if(pq.size()>k)
            {
                pq.pop();
            }
        }
        vector<int> ans(k,0);
        for(int i=(k-1);i>=0;i--)
        {
            ans[i]=(pq.top()).first;
            pq.pop();
        }
        return ans;
    }
};
```

**要注意，sort时，左小于右的compare为从小到大，而在priority_queue中左大于右的compare为小顶堆。**

* 用迭代的方式实现中序遍历

**重点**：要理解遍历停止的条件，即当前遍历的节点为空且栈中为空

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        if(!root) 
        {
            vector<int> v;
            return v;
        }
        vector<int> ans;
        TreeNode* cur=root;
        stack<TreeNode*> st;
        st.push(cur);
        cur=cur->left;
        //栈为空且当前遍历的点为空，才停止中序遍历
        while(cur||!st.empty())//这一步理解很关键，cur表示当前遍历到的点，栈是用于回溯遍历过的点。
        {
            if(cur)
            {
                st.push(cur);
                cur=cur->left;
            }
            else
            {
                cur=st.top();
                st.pop();
                ans.push_back(cur->val);
                cur=cur->right;
            }
        }
        return ans;
    }
};
```

* 迭代法实现后序遍历

**思路** ：参考前序遍历的方式，前序为中左右，后序为左右中，则先更改入栈的顺序，实现中右左的遍历顺序，再翻转结果数组。

实现的代码如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        if(!root)
        {
            vector<int> v;
            return v;
        }
        vector<int> ans;
        TreeNode* cur=root;
        stack<TreeNode*> st;
        st.push(cur);
        while(!st.empty())
        {
            cur=st.top();
            ans.push_back(cur->val);
            st.pop();
            if(cur->left)
                st.push(cur->left);
            if(cur->right)
                st.push(cur->right);
        }
        reverse(ans.begin(),ans.end());
        return ans;
    }
};
```

* 对称二叉树

**思路** ：若要用的递归的方法，则运用类似后序遍历的方式，得到的代码如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool traverse(TreeNode* left,TreeNode* right)
    {
        if((!left)&&(!right)) return true;
        else if(left&&right)
        {
            if(left->val!=right->val) return false;
        }
        else if((!left)&&right) return false;
        else if((!right)&&left) return false;
        bool flag1 = traverse(left->left,right->right);
        bool flag2=traverse(left->right,right->left);
        return flag1&&flag2;
    }
    bool isSymmetric(TreeNode* root) {
        if(!root) return true;
        return traverse(root->left,root->right);
    }
};
```

* 完全二叉树的个数

**思路**：比起一般求二叉树个数，完全二叉树的遍历求个数有下面两种情况：

1. 如果根节点的左子树深度等于右子树深度，则说明 **左子树为满二叉树**
2. 如果根节点的左子树深度大于右子树深度，则说明 **右子树为满二叉树**

比起一般的求二叉树的个数的方法，时间复杂度为O(N)，该方法为O(logN*logN)

可以得到代码如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int countNodes(TreeNode* root) {
        if(!root) return 0;
        int leftdepth=0;
        int rightdepth=0;
        TreeNode* leftnode=root->left;
        TreeNode* rightnode=root->right;
        while(leftnode)
        {
            leftdepth++;
            leftnode=leftnode->left;
        }
        while(rightnode)
        {
            rightdepth++;
            rightnode=rightnode->left;
        }
        if(leftdepth==rightdepth)
            return countNodes(root->right)+(1<<leftdepth);
        else
            return countNodes(root->left)+(1<<rightdepth);
    }
};
```

* 平衡二叉树

给定一个二叉树，判断它是否是高度平衡的二叉树。

本题中，一棵高度平衡二叉树定义为：

> 一个二叉树*每个节点* 的左右两个子树的高度差的绝对值不超过 1 。

**思路**： 参考求二叉树最大深度的递归方法，修改一处逻辑，若回溯到某处，已经高度绝对值之差比1大，此时返回-1，上面的回溯部分均返回-1。

得到的代码如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int getDepth(TreeNode* node)
    {
        if(!node) return 0;
        int leftdepth=getDepth(node->left);
        int rightdepth=getDepth(node->right);
        if(leftdepth==-1||rightdepth==-1) return -1;
        if(abs(leftdepth-rightdepth)>1) return -1;
        return max(leftdepth,rightdepth)+1;
    }
    bool isBalanced(TreeNode* root) {
        if(getDepth(root)==-1) return false;
        return true;
    }
};
```

* 中序遍历和后序遍历构建二叉树

**思路**：主要是递归函数怎么写，构造二叉树时，可以使用返回节点的方式进行递归构建。代码如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    //当要进行构建时，可以运用返回值的方式进行递归
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        if(inorder.size()==0) return NULL;
        TreeNode* node = new TreeNode();
        if(inorder.size()==1)
        {
            node->val=inorder[0];
            return node;
        }
        node->val = postorder[postorder.size()-1];
        int i;
        for(i=0;i<inorder.size();i++)
        {
            if(inorder[i]==node->val)
                break;
        }
        vector<int> leftinorder(inorder.begin(),inorder.begin()+i);
        vector<int> rightinorder(inorder.begin()+i+1,inorder.end());
        vector<int> leftpostoder(postorder.begin(),postorder.begin()+i);
        vector<int> rightpostoder(postorder.begin()+i,postorder.end()-1);
        node->left=buildTree(leftinorder,leftpostoder);
        node->right=buildTree(rightinorder,rightpostoder);
        return node;
    }
};
```

* 二叉搜索树判断

给你一个二叉树的根节点 `root` ，判断其是否是一个有效的二叉搜索树。

**有效** 二叉搜索树定义如下：

- 节点的左子树只包含 **小于** 当前节点的数。
- 节点的右子树只包含 **大于** 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树。

**思路：**注意二叉搜索树的判断可以用中序遍历来判断，同时要注意不能只是单单比较左右孩子节点的值！！！

代码如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool inordertraverse(TreeNode* cur)
    {
        if(!cur) return true;
        if(!inordertraverse(cur->left)) return false;
        if(v.size()==0) v.push_back(cur->val);
        else
        {
            if(v[v.size()-1]>=cur->val) return false;
            else v.push_back(cur->val);
        }
        if(!inordertraverse(cur->right)) return false;
        return true;
    }
    bool isValidBST(TreeNode* root) {
    v.clear();
    return inordertraverse(root);
    }
private:
    vector<int> v;
};
```

* 二叉搜索树的最小绝对差

给你一个二叉搜索树的根节点 `root` ，返回 **树中任意两不同节点值之间的最小差值** 。

差值是一个正数，其数值等于两值之差的绝对值。

**思路：**要搞清楚二叉搜索树的定义！！！不能简单地认为就这个最小值就存在于连续两层之间！！！

代码实现如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void inordertraverse(TreeNode* cur)
    {
        if(!cur) return;
        inordertraverse(cur->left);
        if(pre==-1)
        {
            pre=cur->val;
        }
        else
        {
            judgeval=min(judgeval,cur->val-pre);
            pre=cur->val;
        }
        inordertraverse(cur->right);
    }
    int getMinimumDifference(TreeNode* root) {
        judgeval=1e5+10;
        pre=-1;
        inordertraverse(root);
        return judgeval;
    }
private:
    int judgeval;
    int pre;
};
```

* 二叉搜索树中的众数

给你一个含重复值的二叉搜索树（BST）的根节点 `root` ，找出并返回 BST 中的所有 [众数](https://baike.baidu.com/item/众数/44796)（即，出现频率最高的元素）。

如果树中有不止一个众数，可以按 **任意顺序** 返回。

假定 BST 满足如下定义：

- 结点左子树中所含节点的值 **小于等于** 当前节点的值
- 结点右子树中所含节点的值 **大于等于** 当前节点的值
- 左子树和右子树都是二叉搜索树

**思路：**要想要不借助额外空间，还是采取中序遍历的方式，用pre记录前一个值，并进行比较。代码如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    void inordertraverse(TreeNode* cur)
    {
        if(!cur) return;
        inordertraverse(cur->left);
        if(pre==nullptr)
        {
            pre=cur;
            count=1;
        }
        else
        {
            if(pre->val==cur->val)
            {
                count++;
            }
            else
                count=1;
            pre=cur;
        }
        if(count>maxcount)
        {
            maxcount=count;
            res.clear();
            res.push_back(cur->val);
        }
        else if(count==maxcount)
        {
            res.push_back(cur->val);
        }
        inordertraverse(cur->right);
    }
    vector<int> findMode(TreeNode* root) {
        res.clear();
        pre=nullptr;
        count=maxcount=0;
        inordertraverse(root);
        return res;
    }
private:
    vector<int> res;
    TreeNode* pre;
    int count;
    int maxcount;
};
```

* 二叉树的最近公共祖先

给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

**思路：**按照前序的方式进行遍历（选用前序的原因是这种遍历顺序是自上向下的），而回溯的顺序自底向上（也就是说，遍历到底后，向上回溯），回溯的过程中返回最近公共祖先，nullptr，p或者q（运用类似并查集的思想）。

实现的代码如下：

```c++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(root==nullptr||root==p||root==q) return root;
        TreeNode* lt = lowestCommonAncestor(root->left,p,q);
        TreeNode* rt = lowestCommonAncestor(root->right,p,q);
        //if(root==p||root==q) return root;
        if(lt==nullptr&&rt==nullptr) return nullptr;
        else if(lt&&rt) return root;
        else if(lt) return lt;
        else return rt;
    }
};
```


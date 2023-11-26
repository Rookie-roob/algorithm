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

* 二叉搜索树的最近公共祖先

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

[百度百科](https://baike.baidu.com/item/最近公共祖先/8918834?fr=aladdin)中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（**一个节点也可以是它自己的祖先**）。”

例如，给定如下二叉搜索树: root = [6,2,8,0,4,7,9,null,null,3,5]

**思路：**结合二叉搜索树的性质，自上向下遍历到一个值在p和q对应的值之间或者和其中一个值相等时，立刻停止递归。

代码如下：

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
        if(root==nullptr) return root;
        if(root->val>p->val&&root->val>q->val)
        {
            return lowestCommonAncestor(root->left,p,q);
        }
        else if(root->val<p->val&&root->val<q->val)
        {
            return lowestCommonAncestor(root->right,p,q);
        }
        return root;
    }
};
```

* 删除二叉搜索树中的节点

给定一个二叉搜索树的根节点 **root** 和一个值 **key**，删除二叉搜索树中的 **key** 对应的节点，并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。

一般来说，删除节点可分为两个步骤：

1. 首先找到需要删除的节点；
2. 如果找到了，删除它。

**思路：**类似插入节点时的递归方式，以返回节点的方式进行递归，由于二叉搜索树的性质，只需要递归一边，当遍历到目标节点时，进行分情况讨论：

1. 叶子节点，返回nullptr
2. 左右一边为空，返回另一边即可
3. 左右孩子均非空，将左子树，移到右孩子的最左边。

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
    TreeNode* deleteNode(TreeNode* root, int key) {
        if(root==nullptr) return root;
        if(root->val==key)
        {
            if(root->left==nullptr&&root->right==nullptr)
            {
                delete root;
                return nullptr;
            }
            else if(root->left==nullptr)
            {
                TreeNode* tmp = root->right;
                delete root;
                return tmp;
            }
            else if(root->right==nullptr)
            {
                TreeNode* tmp = root->left;
                delete root;
                return tmp;
            }
            else
            {
                TreeNode* leftchild = root->left;
                TreeNode* rightchild = root->right;
                while(rightchild->left)
                {
                    rightchild=rightchild->left;
                }
                rightchild->left=leftchild;
                rightchild = root->right;
                delete root;
                return rightchild;
            }
        }
        else if(root->val>key)
        {
            root->left=deleteNode(root->left,key);
        }
        else
        {
            root->right=deleteNode(root->right,key);
        }
        return root;
    }
};
```

* 修剪二叉搜索树

给你二叉搜索树的根节点 `root` ，同时给定最小边界`low` 和最大边界 `high`。通过修剪二叉搜索树，使得所有节点的值在`[low, high]`中。修剪树 **不应该** 改变保留在树中的元素的相对结构 (即，如果没有被移除，原有的父代子代关系都应当保留)。 可以证明，存在 **唯一的答案** 。

所以结果应当返回修剪好的二叉搜索树的新的根节点。注意，根节点可能会根据给定的边界发生改变。

**思路：**类似插入和删除二叉搜索树的节点，利用返回节点的方式进行递归回溯，若当前节点小于low，则不断用其右孩子进行代替，以此类推，除非遍历到了空节点；当前节点大于high的情况与上面类似。最终得到的代码如下：

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
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        if(!root) return root;
        if(root->val<low)
        {
            return trimBST(root->right,low,high);
        }
        if(root->val>high)
        {
            return trimBST(root->left,low,high);
        }
        root->left = trimBST(root->left,low,high);
        root->right = trimBST(root->right,low,high);
        return root;
    }
};
```

* 电话号码的字母组合

  给定一个仅包含数字 `2-9` 的字符串，返回所有它能表示的字母组合。答案可以按 **任意顺序** 返回。

  给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。

  ![img](picture/phone_number.png)

**思路：**这题主要的点在于，每个数字对应的字符串是必须要选的，是不可以跳过去的，在遍历的时候一定要注意！！！

代码如下：

```c++
class Solution {
public:
    void backtrace(string& s,string digits,int index)
    {
        if(index==digits.length())
        {
            ans.push_back(s);
            return;
        }
        int i=index;
        for(int j=0;j<(nummap[digits[i]-'0']).size();j++)
        {
            s = s+nummap[digits[i]-'0'][j];
            backtrace(s,digits,i+1);
            s.resize(s.length()-1);
        }
    }
    vector<string> letterCombinations(string digits) {
        ans.clear();
        if(digits.size()==0) return ans;
        nummap[2] = vector<string>{"a","b","c"};
        nummap[3] = vector<string>{"d","e","f"};
        nummap[4] = vector<string>{"g","h","i"};
        nummap[5] = vector<string>{"j","k","l"};
        nummap[6] = vector<string>{"m","n","o"};
        nummap[7] = vector<string>{"p","q","r","s"};
        nummap[8] = vector<string>{"t","u","v"};
        nummap[9] = vector<string>{"w","x","y","z"};
        string s;
        backtrace(s,digits,0);
        return ans;
    }
private:
    unordered_map<int,vector<string>> nummap;
    vector<string> ans;
};
```

* 复原IP地址

**有效 IP 地址** 正好由四个整数（每个整数位于 `0` 到 `255` 之间组成，且不能含有前导 `0`），整数之间用 `'.'` 分隔。

- 例如：`"0.1.2.201"` 和` "192.168.1.1"` 是 **有效** IP 地址，但是 `"0.011.255.245"`、`"192.168.1.312"` 和 `"192.168@1.1"` 是 **无效** IP 地址。

给定一个只包含数字的字符串 `s` ，用以表示一个 IP 地址，返回所有可能的**有效 IP 地址**，这些地址可以通过在 `s` 中插入 `'.'` 来形成。你 **不能** 重新排序或删除 `s` 中的任何数字。你可以按 **任何** 顺序返回答案。

**思路：**这里一开始没有搞清楚的点是，遇到了前缀0，直接加入一个0，但是要注意的是，加入后要进行回溯！！！这一步不能忘记，否则就会造成其他地方回溯弹出了这个0，那么就会发生错误！！！

代码如下：

```c++
class Solution {
public:
    int tonum(string& s)
    {
        int num=0;
        for(int i=0;i<s.length();i++)
            num=(num*10+s[i]-'0');
        return num;
    }
    void backtrace(string& s,vector<string>& v,int index)
    {
        if(index==s.length()&&v.size()==4)
        {
            ans.push_back(v);
            return;
        }
        else if(index==s.length()||v.size()==4)
            return;
        else
        {
            if(s[index]=='0')
            {
                v.push_back("0");
                backtrace(s,v,index+1);
                v.pop_back();
            }
            else
            {
                for(int i=index;(i<s.length())&&((i-index)<3);i++)
                {
                    string subs = s.substr(index,i-index+1);
                    int num = tonum(subs);
                    if(num>=0&&num<=255)
                    {
                        v.push_back(subs);
                        backtrace(s,v,i+1);
                        v.pop_back();
                    }
                }
            }
        }
    }
    vector<string> restoreIpAddresses(string s) {
        vector<string> v1;
        if(s.length()<4||s.length()>12) return v1;
        ans.clear();
        vector<string> v;
        backtrace(s,v,0);
        
        for(int i=0;i<ans.size();i++)
        {
            string s1=ans[i][0];
            for(int j=1;j<4;j++)
            {
                s1+=".";
                s1+=ans[i][j];
            }
            v1.push_back(s1);
        }
        return v1;
    }
private:
    vector<vector<string>> ans;
};
```

* 含重复数字的重排列

给定一个可包含重复数字的序列 `nums` ，***按任意顺序*** 返回所有不重复的全排列。

**思路：**活用used数组进行遍历回溯，但是要注意先进行排序！

代码如下：

```c++
class Solution {
public:
    void backtrace(vector<int>& v,vector<int>& nums,vector<bool>& used)
    {
        if(v.size()==nums.size())
        {
            ans.push_back(v);
            return;
        }
        for(int i=0;i<nums.size();i++)
        {
            if(i>0&&nums[i]==nums[i-1]&&used[i-1]==false)
                continue;
            if(used[i]==false)
            {
                used[i]=true;
                v.push_back(nums[i]);
                backtrace(v,nums,used);
                v.pop_back();
                used[i]=false;
            }
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        ans.clear();
        vector<int> v;
        vector<bool> used(nums.size(),false);
        sort(nums.begin(),nums.end());
        backtrace(v,nums,used);
        return ans;
    }
private:
    vector<vector<int>> ans;
};
```

实现方式还有下面的方式，但是较为麻烦，memset其实也要耗费时间！！！

代码如下：

```c++
int a[21];
class Solution {
public:
    void backtrace(vector<int>& v,vector<int>& nums)
    {
        if(v.size()==nums.size())
        {
            ans.push_back(v);
            return;
        }
        bool judge[21];
        memset(judge,false,sizeof(judge));
        for(int i=0;i<nums.size();i++)
        {
            if(a[nums[i]+10]&&judge[nums[i]+10]==false)
            {
                judge[nums[i]+10]=true;
                a[nums[i]+10]--;
                v.push_back(nums[i]);
                backtrace(v,nums);
                v.pop_back();
                a[nums[i]+10]++;
            }
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        ans.clear();
        vector<int> v;
        memset(a,0,sizeof(a));
        for(int i=0;i<nums.size();i++)
            a[nums[i]+10]++;
        backtrace(v,nums);
        return ans;
    }
private:
    vector<vector<int>> ans;
};
```

还是觉得第一种方法更好！！！

* 重新安排行程

给你一份航线列表 `tickets` ，其中 `tickets[i] = [fromi, toi]` 表示飞机出发和降落的机场地点。请你对该行程进行重新规划排序。

所有这些机票都属于一个从 `JFK`（肯尼迪国际机场）出发的先生，所以该行程必须从 `JFK` 开始。如果存在多种有效的行程，请你按字典排序返回最小的行程组合。

- 例如，行程 `["JFK", "LGA"]` 与 `["JFK", "LGB"]` 相比就更小，排序更靠前。

假定所有机票至少存在一种合理的行程。且所有的机票 必须都用一次 且 只能用一次。

**思路：**要注意，这个题相同的机票可能会有多张！！！且可能会在遍历过程中到一个死胡同，此时是需要回溯的！！！！而且可能会有A->B，B->A这样的循环路径存在，因此需要一个很好的数据结构进行记录，选用unordered_map<string,map<string,int>>，因为遍历过程中，是需要删除操作的，不然遍历无法向下进行，而单单地选用unordered_map<string,multi_set<string>>，在遍历set时，是无法删除值的，不然迭代器会无效！！！

代码如下：

```c++
class Solution {
public:
    bool backtrace(string source,vector<string>& v)
    {
        if(v.size()==total_count)
        {
            ans=v;
            return true;
        }
        for(auto iter=(mp[source]).begin();iter!=(mp[source]).end();iter++)
        {
            if((*iter).second==0)
                continue;
            else
            {
                v.push_back((*iter).first);
                (*iter).second--;
                if(backtrace((*iter).first,v))
                {
                    return true;
                }
                v.pop_back();
                (*iter).second++;
            }
        }
        return false;
    }
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        mp.clear();
        ans.clear();
        total_count=1;
        for(int i=0;i<tickets.size();i++)
        {
            mp[tickets[i][0]][tickets[i][1]]++;
            total_count++;
        }
        vector<string> v;
        v.push_back("JFK");
        backtrace("JFK",v);
        return ans;
    }
private:
    unordered_map<string,map<string,int>> mp;
    vector<string> ans;
    int total_count;
};
```

* 摆动序列

如果连续数字之间的差严格地在正数和负数之间交替，则数字序列称为 **摆动序列 。**第一个差（如果存在的话）可能是正数或负数。仅有一个元素或者含两个不等元素的序列也视作摆动序列。

- 例如， `[1, 7, 4, 9, 2, 5]` 是一个 **摆动序列** ，因为差值 `(6, -3, 5, -7, 3)` 是正负交替出现的。
- 相反，`[1, 4, 7, 2, 5]` 和 `[1, 7, 4, 5, 5]` 不是摆动序列，第一个序列是因为它的前两个差值都是正数，第二个序列是因为它的最后一个差值为零。

**子序列** 可以通过从原始序列中删除一些（也可以不删除）元素来获得，剩下的元素保持其原始顺序。

给你一个整数数组 `nums` ，返回 `nums` 中作为 **摆动序列** 的 **最长子序列的长度** 。

**思路：**当(preDiff <= 0 && curDiff > 0) || (preDiff >= 0 && curDiff < 0)时，说明到了一个波峰或者波谷，这里我们在处理的过程中，遇到了平地时，会取最右边的情况。我们假定最右边固定有一个波峰或者波谷，还要注意“上-平-上”这样的情况下只能记录一个波峰！！！

```c++
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int prediff = 0;
        int curdiff;
        int cnt=1;
        for(int i=0;i<(nums.size()-1);i++)
        {
            curdiff=nums[i+1]-nums[i];
            if((prediff<=0&&curdiff>0)||(prediff>=0&&curdiff<0))
            {
                cnt++;
                prediff=curdiff;//这一步要在确定要转换波峰波谷的时候进行代换！！！
            }
        }
        return cnt;
    }
};
```

* 跳跃游戏

给你一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

 **思路：**注意本题只是要求得出可不可行，所以可以按照cover覆盖范围来求解，如果覆盖范围变大至数组边界，则有解，若遍历到了cover边界，该cover还是没有动，那么无解。当然也可以用现在覆盖，下一个覆盖的方式来求解（下一个题目）

代码如下：

A.

```c++
class Solution {
public:
    //这道题只要求得到能不能求出解
    bool canJump(vector<int>& nums) {
        if(nums.size()==1) return true;
        int cover=0;
        for(int i=0;i<=cover;i++)
        {
            if((nums[i]+i)>cover)
            {
                cover=nums[i]+i;
                if(cover>=(nums.size()-1)) return true;
            }
        }
        return false;
    }
};
```



B.

```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int point=0;
        int area = nums[0];
        while(!(((point+area)>=(nums.size()-1))||(area==0)))
        {
            //cout<<point<<" "<<area<<endl;
            int tmp=0;
            int p=point;
            for(int i=point;i<=(area+point)&&i<nums.size();i++)
            {
                if((i+nums[i])>tmp)
                {
                    tmp=(i+nums[i]);
                    p=i;
                }
                //cout<<"next"<<" "<<i<<" "<<nums[i]<<endl;
            }
            //cout<<"last"<<" "<<p<<" "<<nums[p]<<endl;
            if(tmp==(point+area)&&tmp<(nums.size()-1))
                return false;
            point=p;
            area=nums[point];
            //cout<<"next loop "<<point<<" "<<area<<endl;
        }
        if((point+area)>=(nums.size()-1))
            return true;
        else
            return false;
    }
};
```

* 跳跃游戏2

给定一个长度为 `n` 的 **0 索引**整数数组 `nums`。初始位置为 `nums[0]`。

每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:

- `0 <= j <= nums[i]` 
- `i + j < n`

返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。

**思路：**用当前覆盖以及下一次覆盖进行求解，要注意代码写的简洁性。

代码如下：

A.

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        if(nums.size()==1) return 0;
        int curdist = 0;
        int nextdist = 0;
        int cnt = 0;
        for(int i=0;i<(nums.size()-1);i++)
        {
            nextdist=max(nextdist,i+nums[i]);
            if(i==curdist)
            {
                cnt++;
                curdist=nextdist;
            }
        }
        return cnt;
    }
};
```

B.

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        if(nums.size()==1) return 0;
        int point=0;
        int area = nums[0];
        int cnt=1;
        while(!(((point+area)>=(nums.size()-1))||(area==0)))
        {
            //cout<<point<<" "<<area<<endl;
            cnt++;
            int tmp=0;
            int p=point;
            for(int i=point;i<=(area+point)&&i<nums.size();i++)
            {
                if((i+nums[i])>tmp)
                {
                    tmp=(i+nums[i]);
                    p=i;
                }
                //cout<<"next"<<" "<<i<<" "<<nums[i]<<endl;
            }
            //cout<<"last"<<" "<<p<<" "<<nums[p]<<endl;
            if(tmp==(point+area)&&tmp<(nums.size()-1))
                return -1;
            point=p;
            area=nums[point];
            //cout<<"next loop "<<point<<" "<<area<<endl;
        }
        if((point+area)>=(nums.size()-1))
            return cnt;
        else
            return -1;
    }
};
```

* K次取反后最大化的数组和

给你一个整数数组 `nums` 和一个整数 `k` ，按以下方法修改该数组：

- 选择某个下标 `i` 并将 `nums[i]` 替换为 `-nums[i]` 。

重复这个过程恰好 `k` 次。可以多次选择同一个下标 `i` 。

以这种方式修改数组后，返回数组 **可能的最大和** 。

**思路：**这里其实要注意按照下面的写法，当把负的转换成正的后，此时可能k可能还有剩余，根据k的奇偶判断是否要转换当前的元素。

代码如下：

```c++
class Solution {
public:
    int largestSumAfterKNegations(vector<int>& nums, int k) {
        sort(nums.begin(),nums.end());
        int index=0;
        while(k--)
        {
            if(nums[index]<0)
            {
                nums[index]=-nums[index];
                index++;
                if(index==nums.size())//这一步不能忘记，此时是会到达数组结尾的
                {
                    if(k%2==0) break;
                    else
                    {
                        nums[index-1]=-nums[index-1];
                        break;
                    }
                }
            }
            else if(nums[index]==0)
            {
                break;
            }
            else
            {
                if(k%2==1) break;
                else
                {
                    if(index==0) nums[index]=-nums[index];
                    else if(index>0&&nums[index]>nums[index-1]) nums[index-1]=-nums[index-1];
                    else nums[index]=-nums[index];
                    break;
                }
            }
        }
        int tmpsum=0;
        for(auto v:nums)
            tmpsum+=v;
        return tmpsum;
    }
};
```

思路更简洁的方式如下：（先将数组按照绝对值从大到小排序，而后从前到后遍历，碰到负数且k>0，此时取反。而后如果k还有剩余，则对于绝对值最小的进行后续判断）

```c++
class Solution {
public:
    static bool cmp(const int& a,const int& b)
    {
        return abs(a)>abs(b);
    }
    int largestSumAfterKNegations(vector<int>& nums, int k) {
        sort(nums.begin(),nums.end(),cmp);
        for(int i=0;i<nums.size();i++)
        {
            if(k>0&&nums[i]<0)
            {
                nums[i]=-nums[i];
                k--;
            }
        }
        if(k>0)
        {
            if(k%2==1)
                nums[nums.size()-1]=-nums[nums.size()-1];
        }
        int tmp=0;
        for(auto v:nums)
            tmp+=v;
        return tmp;
    }
};
```

* 加油站（重点思考题目！！！）

在一条环路上有 `n` 个加油站，其中第 `i` 个加油站有汽油 `gas[i]` 升。

你有一辆油箱容量无限的的汽车，从第 `i` 个加油站开往第 `i+1` 个加油站需要消耗汽油 `cost[i]` 升。你从其中的一个加油站出发，开始时油箱为空。

给定两个整数数组 `gas` 和 `cost` ，如果你可以按顺序绕环路行驶一周，则返回出发时加油站的编号，否则返回 `-1` 。如果存在解，则 **保证** 它是 **唯一** 的。

**思路：**从i到j如果开不到，则只能从j+1开始考虑。这个题主要难论证的点是：怎么证明解是有效的，最后的想法是用类似反证法的方式，最后得到一个点k，k到末尾的cursum是大于0的，如果开头有段距离cursum小于0，此时后半段肯定要有相应的补回来，因为总的totalsum是大于0的，以此类推，如果中间又有一段是小于0，那么后面还要继续补回来。。。直到最后到达最终的点。

代码如下（整体的代码在这个思路下还是很好写的，主要是这个思路！！！）：

```c++
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int totalsum =  0;
        int cursum = 0;
        int index=0;
        for(int i=0;i<gas.size();i++)
        {
            totalsum+=gas[i]-cost[i];
            cursum+=gas[i]-cost[i];
            if(cursum<0)
            {
                cursum=0;
                index=(i+1);
            }
        }
        if(totalsum<0) return -1;
        return index;
    }
};
```

* 分发糖果

`n` 个孩子站成一排。给你一个整数数组 `ratings` 表示每个孩子的评分。

你需要按照以下要求，给这些孩子分发糖果：

- 每个孩子至少分配到 `1` 个糖果。
- 相邻两个孩子评分更高的孩子会获得更多的糖果。

请你给每个孩子分发糖果，计算并返回需要准备的 **最少糖果数目** 。

**思路：**这题可以想到既要从前向后看，也要从后往前看。主要是在做的时候，注意想到先将分发的糖果都初始化为1，可以做两次做遍历，做两次贪心，第二次的时候取**最大值**。

代码如下：

```c++
class Solution {
public:
    int candy(vector<int>& ratings) {
        vector<int> v(ratings.size(),1);
        
        for(int i=1;i<ratings.size();i++)
        {
            if(ratings[i]>ratings[i-1])
                v[i]=v[i-1]+1;
        }

        for(int i=(ratings.size()-2);i>=0;i--)
        {
            if(ratings[i]>ratings[i+1])
            {
                v[i]=max(v[i],v[i+1]+1);
            }
        }
        int sum=0;
        for(auto s:v)
            sum+=s;
        return sum;
    }
};
```

* 根据身高重建队列

假设有打乱顺序的一群人站成一个队列，数组 `people` 表示队列中一些人的属性（不一定按顺序）。每个 `people[i] = [hi, ki]` 表示第 `i` 个人的身高为 `hi` ，前面 **正好** 有 `ki` 个身高大于或等于 `hi` 的人。

请你重新构造并返回输入数组 `people` 所表示的队列。返回的队列应该格式化为数组 `queue` ，其中 `queue[j] = [hj, kj]` 是队列中第 `j` 个人的属性（`queue[0]` 是排在队列前面的人）。

**思路：**把这个队伍重建的过程想象成把人一个个插入队列的过程。按照身高从高到低，若身高相等则按照ki从小到大进行排序，按照这个顺序进行插入，可以保证这个插入过程是正确的，因为身高低的后插入，而比他大的之前就在队列里了，身高相等的，也是按照顺序进行插入（因为ki表示前面有ki个大于或等于 `hi`的人，所以身高相等则按照ki从小到大进行排序）。

这里要注意写代码时，要用到vector的insert，做到动态插入；同时，在写排序的cmp函数时，不能简单地写a>b||c\<d这种形式，因为这样的话并不能表示相等时判断c，d，不相等时判断a，b，要写成a==b?c\<d:a\>b，写代码时要仔细！！！！

```c++
class Solution {
public:
    static bool cmp(const vector<int>& v1,const vector<int>& v2)
    {
        return v1[0]==v2[0]?v1[1]<v2[1]:v1[0]>v2[0];
    }
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort(people.begin(),people.end(),cmp);
        vector<vector<int>> ans;
        for(int i=0;i<people.size();i++)
        {
            int pos = people[i][1];
            ans.insert(ans.begin()+pos,people[i]);
        }
        return ans;
    }
};
```

* 单调递增的数字

当且仅当每个相邻位数上的数字 `x` 和 `y` 满足 `x <= y` 时，我们称这个整数是**单调递增**的。

给定一个整数 `n` ，返回 *小于或等于 `n` 的最大数字，且数字呈 **单调递增*** 。

 

**示例 1:**

```
输入: n = 10
输出: 9
```

**示例 2:**

```
输入: n = 1234
输出: 1234
```

**示例 3:**

```
输入: n = 332
输出: 299
```

 

**提示:**

- `0 <= n <= 109`

**思路：**暴力法很容易超时。可以从前向后或者从后向前逐位遍历，这里采用从后向前比较好，因为从前向后遍历的话，遇到逆序的时候，还要回去遍历。主要思想就是遇到逆序的，如12222222221，遇到逆序的21，则考虑后面的位数到替换为9，主要就是要找到从哪一位开始替换为9，这里的话要找到逆序数对的前一个，对于该数向前最多能到哪一位（在这个区间中的数都相等，都为这个逆序数对的前一个），把那个最前的减1即可，对于这个例子，到了1后面的那一位，2-1=1，后面全为9，则最后答案为11999999999。

代码如下：

**从前向后：**

```c++
class Solution {
public:
    int monotoneIncreasingDigits(int n) {
        if(n<10) return n;
        vector<int> v;
        int ncopy=n;
        while(n)
        {
            v.push_back(n%10);
            n=n/10;
        }
        vector<int> ans;
        bool flag=false;
        for(int i=(v.size()-2);i>=0;i--)
        {
            if(v[i]>=v[i+1])
            {
                ans.push_back(v[i+1]);
            }
            else
            {
                flag=true;
                if(((i+2)<v.size())&&v[i+1]==v[i+2])
                {
                    int index = -1;
                    for(int j=(ans.size()-1);j>=0;j--)
                    {
                        if(ans[j]!=v[i+1])
                        {
                            index=j;
                            break;
                        }
                    }
                    //index+1到ans.size()-1值相等
                    ans[index+1]--;
                    for(int j=index+2;j<ans.size();j++)
                        ans[j]=9;
                    for(int j=(i+1);j>=0;j--)
                    {
                        ans.push_back(9);
                    }
                }
                else
                {
                    ans.push_back(v[i+1]-1);
                    for(int j=i;j>=0;j--)
                    {
                        ans.push_back(9);
                    }
                }
                break;
            }
        }
        if(!flag) return ncopy;
        else
        {
            int num=0;
            for(auto item:ans)
                num=num*10+item;
            return num;
        }
    }
};
```



**从后向前：**

```c++
class Solution {
public:
    int monotoneIncreasingDigits(int n) {
        //从后向前遍历会更加直观简便，从前向后遍历还要往回看，很麻烦
        string s = to_string(n);//转换成string，这一步很巧妙
        int flag=-1;
        for(int i=s.length()-1;i>0;i--)
        {
            if(s[i]<s[i-1])
            {
                s[i-1]--;
                flag=i-1;
            }
        }
        if(flag==-1) return n;
        else
        {
            for(int i=flag+1;i<s.length();i++)
            {
                s[i]='9';
            }
            return stoi(s);
        }
    }
};
```

* 划分字母区间

给你一个字符串 `s` 。我们要把这个字符串划分为尽可能多的片段，同一字母最多出现在一个片段中。

注意，划分结果需要满足：将所有划分结果按顺序连接，得到的字符串仍然是 `s` 。

返回一个表示每个字符串片段的长度的列表。

**思路：**该题可以采用最多互不相交的集合来解决，集合之间互不相交。当然更巧妙的是通过记录每个字母最多能到哪一位，然后，从前向后遍历，如果这个区间中每个字母的最远出现范围都被包在这个遍历的区间中，则找到了一个字母区间。

代码如下：

**最多互不相交的集合：**

```c++
class Solution {
public:
    static bool cmp(vector<int>& x,vector<int>& y)
    {
        return x[0]<y[0];
    }
    vector<int> partitionLabels(string s) {
        memset(a,-1,sizeof(a));
        for(int i=0;i<s.length();i++)
        {
            if(a[s[i]-'a'][0]==-1)
            {
                a[s[i]-'a'][0]=i;
                a[s[i]-'a'][1]=i;
            }
            else
            {
                a[s[i]-'a'][1]=i;
            }
        }
        vector<vector<int>> v;
        for(int i=0;i<26;i++)
        {
            if(a[i][0]!=-1)
            {
                v.push_back({a[i][0],a[i][1]});
            }
        }
        sort(v.begin(),v.end(),cmp);
        vector<int> ans;
        int rmax;
        int lmin;
        for(int i=0;i<v.size();i++)
        {
            if(i==0)
            {
                rmax=v[i][1];
                lmin=v[i][0];
            }
            else
            {
                if(v[i][0]>rmax)
                {
                    ans.push_back(rmax-lmin+1);
                    lmin=v[i][0];
                    rmax=v[i][1];
                }
                else
                {
                    rmax=max(v[i][1],rmax);
                }
            }
        }
        ans.push_back(rmax-lmin+1);
        return ans;


    }
private:
    int a[26][2];
};
```

**巧妙的思路：**

```c++
class Solution {
public:
    static bool cmp(vector<int>& x,vector<int>& y)
    {
        return x[0]<y[0];
    }
    vector<int> partitionLabels(string s) {
        memset(a,-1,sizeof(a));
        for(int i=0;i<s.length();i++)
        {
            if(a[s[i]-'a'][0]==-1)
            {
                a[s[i]-'a'][0]=i;
                a[s[i]-'a'][1]=i;
            }
            else
            {
                a[s[i]-'a'][1]=i;
            }
        }
        vector<vector<int>> v;
        for(int i=0;i<26;i++)
        {
            if(a[i][0]!=-1)
            {
                v.push_back({a[i][0],a[i][1]});
            }
        }
        sort(v.begin(),v.end(),cmp);
        vector<int> ans;
        int rmax;
        int lmin;
        for(int i=0;i<v.size();i++)
        {
            if(i==0)
            {
                rmax=v[i][1];
                lmin=v[i][0];
            }
            else
            {
                if(v[i][0]>rmax)
                {
                    ans.push_back(rmax-lmin+1);
                    lmin=v[i][0];
                    rmax=v[i][1];
                }
                else
                {
                    rmax=max(v[i][1],rmax);
                }
            }
        }
        ans.push_back(rmax-lmin+1);
        return ans;


    }
private:
    int a[26][2];
};
```

* 监控二叉树

给定一个二叉树，我们在树的节点上安装摄像头。

节点上的每个摄影头都可以监视**其父对象、自身及其直接子对象。**

计算监控树的所有节点所需的最小摄像头数量。

**思路：**为了避免叶子节点放摄像头，这会造成指数级别的增加，故我们遍历时要从类似自底向上的方式进行遍历，所以采用后序遍历。为了确定该节点需不需要放摄像头，采用状态转换的方式来判断，根据左右节点的状态，判断该节点需不需要放摄像头。状态可分为3类，即放摄像头，能够被覆盖，不能被覆盖。其中当状态为放摄像头时，统计的数目进行增加。

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
    //0：无覆盖    1：放摄像头   2：有覆盖
    int traverse(TreeNode* node)
    {
        if(!node) return 2;
        int lst = traverse(node->left);
        int rst = traverse(node->right);
        if(lst==0||rst==0) 
        {
            totalnum++;
            return 1;
        }
        else if(lst==1||rst==1) return 2;
        else return 0;
    }
    int minCameraCover(TreeNode* root) {
        totalnum=0;
        if(traverse(root)==0) totalnum++;
        return totalnum;
    }
private:
    int totalnum;
};
```

* 目标和

给你一个非负整数数组 `nums` 和一个整数 `target` 。

向数组中的每个整数前添加 `'+'` 或 `'-'` ，然后串联起所有整数，可以构造一个 **表达式** ：

- 例如，`nums = [2, 1]` ，可以在 `2` 之前添加 `'+'` ，在 `1` 之前添加 `'-'` ，然后串联起来得到表达式 `"+2-1"` 。

返回可以通过上述方法构造的、运算结果等于 `target` 的不同 **表达式** 的数目。

**思路：**把这个问题转换成背包问题，直接做的话，空间复杂度很高！！！

```c+++
class Solution {
public:
    //x-(S-x)=2x-S=t
    //x=(S+t)/2
    int findTargetSumWays(vector<int>& nums, int target) {
        int sum=0;
        for(auto num:nums)
            sum+=num;
        if(target>sum) return 0;
        sum+=target;
        if(sum<0) return 0;
        if(sum%2!=0) return 0;
        sum=sum/2;
        dp[0]=1;
        for(int i=0;i<nums.size();i++)
        {
            for(int j=sum;j>=nums[i];j--)
            {
                dp[j]+=dp[j-nums[i]];
            }
        }
        return dp[sum];
    }
private:
    int dp[1001];
};
```

直接做的话代码如下，可以看到空间复杂度很高！！！

```c++
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        dp[0][nums[0]+1000]++;
        dp[0][-nums[0]+1000]++;
        set.insert(nums[0]+1000);
        set.insert(-nums[0]+1000);
        for(int i=1;i<nums.size();i++)
        {
            unordered_set<int> set1;
            for(auto iter=set.begin();iter!=set.end();iter++)
            {
                dp[i][nums[i]+(*iter)]+=dp[i-1][*iter];
                dp[i][-nums[i]+(*iter)]+=dp[i-1][*iter];
                set1.insert(nums[i]+(*iter));
                set1.insert(-nums[i]+(*iter));
            }
            set=set1;
        }
        return dp[nums.size()-1][target+1000];
    }
private:
    int dp[21][2001];
    unordered_set<int> set;
};
```

* 打家劫舍Ⅱ

你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 **围成一圈** ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警** 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **在不触动警报装置的情况下** ，今晚能够偷窃到的最高金额。

**思路：**要把这道题的情况分开来考虑，start到end-1以及start+1到end这两种情况，分别考虑这两种情况的最大值，最后再取最大值即可。

代码如下：

```c++
class Solution {
public:
    int robtraverse(int start,int end,vector<int>& nums)
    {
        if(start==end) return nums[start];
        else if(start==(end-1)) return max(nums[start],nums[end]);
        else
        {
            dp[start]=nums[start];
            dp[start+1]=max(nums[start],nums[start+1]);
            for(int i=(start+2);i<=end;i++)
            {
                dp[i]=max(dp[i-1],dp[i-2]+nums[i]);
            }
            return dp[end];
        }
    }

    int rob(vector<int>& nums) {
        if(nums.size()==1) return nums[0];
        return max(robtraverse(0,nums.size()-2,nums),robtraverse(1,nums.size()-1,nums));
    }
private:
    int dp[101];
};
```

* 打家劫舍Ⅲ

小偷又发现了一个新的可行窃的地区。这个地区只有一个入口，我们称之为 `root` 。

除了 `root` 之外，每栋房子有且只有一个“父“房子与之相连。一番侦察之后，聪明的小偷意识到“这个地方的所有房屋的排列类似于一棵二叉树”。 如果 **两个直接相连的房子在同一天晚上被打劫** ，房屋将自动报警。

给定二叉树的 `root` 。返回 ***在不触动警报的情况下** ，小偷能够盗取的最高金额* 。

**思路：**记忆化搜索很容易想到，主要是还有一种方法为树形dp，在递归过程中维护一个dp数组，有两个元素，分别表示要不要选当前元素，这样的树形dp也能够保证每个点只遍历一次，运用后序遍历即可。

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
    vector<int> treedp(TreeNode* cur)
    {
        if(cur==nullptr) return vector<int>(2,0);
        vector<int> lson = treedp(cur->left);
        vector<int> rson = treedp(cur->right);
        vector<int> ret(2,0);
        ret[0]=max(lson[0],lson[1])+max(rson[0],rson[1]);
        ret[1]=lson[0]+rson[0]+cur->val;
        return ret;
    }
    int rob(TreeNode* root) {
        vector<int> v = treedp(root);
        return max(v[0],v[1]);
    }
};
```

记忆化搜索的代码如下：

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
    int traverse(TreeNode* cur)
    {
        if(!cur) return 0;
        if(cur->left==nullptr&&cur->right==nullptr) return cur->val;
        int lson=0;
        int rson=0;
        int llson,lrson,rlson,rrson;
        llson=lrson=rlson=rrson=0;
        if(cur->left!=nullptr)
        {
            if(map.find(cur->left)!=map.end()) lson=map[cur->left];
            else 
            {
                lson=traverse(cur->left);
                map[cur->left]=lson;
            }
            if(cur->left->left==nullptr) ;
            else if(map.find(cur->left->left)!=map.end()) llson=map[cur->left->left];
            else
            {
                llson=traverse(cur->left->left);
                map[cur->left->left]=llson;
            }
            if(cur->left->right==nullptr) ;
            else if(map.find(cur->left->right)!=map.end()) lrson=map[cur->left->right];
            else
            {
                lrson=traverse(cur->left->right);
                map[cur->left->right]=lrson;
            }
        }
        
        
        if(cur->right!=nullptr)
        {
            if(map.find(cur->right)!=map.end()) rson=map[cur->right];
            else 
            {
                rson=traverse(cur->right);
                map[cur->right]=rson;
            }
            if(cur->right->left==nullptr) ;
            else if(map.find(cur->right->left)!=map.end()) rlson=map[cur->right->left];
            else
            {
                rlson=traverse(cur->right->left);
                map[cur->right->left]=rlson;
            }
            if(cur->right->right==nullptr) ;
            else if(map.find(cur->right->right)!=map.end()) rrson=map[cur->right->right];
            else
            {
                rrson=traverse(cur->right->right);
                map[cur->right->right]=rrson;
            }
        }
        return max(lson+rson,llson+lrson+rlson+rrson+cur->val);
    }
    int rob(TreeNode* root) {
        if(!root) return 0;
        if(root->left==nullptr&&root->right==nullptr) return root->val;
        map.clear();
        int ret = traverse(root);
        return ret;
    }
private:
    unordered_map<TreeNode*,int> map;
};
```


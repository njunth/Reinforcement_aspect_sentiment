#coding=utf-8
import sys
def _numof(str):
    length=len(str)
    dp = [[0 for i in range(length)] for i in range(length)]
    for j in range(length):
        dp[j][j]=1
        i=j-1
        while(i>=0):
            dp[i][j]=dp[i+1][j]+dp[i][j-1]-dp[i+1][j-1]
            if(str[i]==str[j]):
                dp[i][j]+=1+dp[i+1][j-1]
            i-=1
    return dp[0][length-1]
if __name__ == "__main__":
    # 读取第一行的str
    str = raw_input()
    num=_numof(str)
    print num
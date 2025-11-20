test 主要测试不同prompt对划分think的影响

test2 主要测试\n\n  + We can get the question's Final Answer: \\boxed对划分think的影响test

test3 主要在test2的基础上对confidence进行测试

test4 主要是测试通过few-shot让强模型对弱模型的完整输出进行划分

test5 主要是对test4保存的数据进行confidence和续写acc测试

test6 主要是原模型对自己输出的部分结果进行划分

test7 在测试流式输出 + We can get the question's Final Answer: \\boxed是否可以作为替代

test7_para_accuracy 主要是测试在test7的最高confidence上进行切分，续写和原来的准确率比较是否下降。

思路：

每个节点在每次探索都有其info_entropy，value，方案截断节点还有特有的confidence（不变）

value和info_entropy在每次探索会改变

思路1：value和info在每次搜索后都会变化，所以保存探索树，每个节点都保存了不同探索时期的价值和info_entropy数组。

思路2：连续生成，随机生成，生成时不计算info_entropy，读取树的时候才计算。

思路3：生成部分的树，生成的时候计算info_entropy，在后面随机（?）进行生成，不记录entropy?

问题4：value的价值是通过confidence计算的，或者说需要将info_entropy加入到value里面吗？info_entropy和value到底是成正比还是反比？

探索需要平衡token，info_entropy，value？需要平衡的是否有点太多？



1. 怎么写好论文，讲好故事。
2. 怎么想idea。
3. 看论文看到什么程度，它的分析有必要看吗。
4.

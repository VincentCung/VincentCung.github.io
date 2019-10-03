# 使用[NexT]latex 事先必须配置MathJax

参考：[使用LaTex添加公式到Hexo博客里 in 简书](https://www.jianshu.com/p/68e6f82d88b7)

配置`NexT`主题支持数学公式，分为两步

1. 配置`hexo`渲染器
2. 配置`NexT`内部数学公式渲染引擎

## 配置`hexo`渲染器

进入hexo根目录
```
npm un hexo-renderer-marked --save
npm un hexo-renderer-marked-it-plus --save 关键！！
npm un hexo-renderer-marked-it --save # 关键！！！
npm i  hexo-renderer-kramed --save
```

## 配置NexT内部数学公式渲染引擎

进入`themes/next/_config.yml`，找到`math`配置

```# Math Formulas Render Support
math:
  enable: true

  # Default (true) will load mathjax / katex script on demand.
  # That is it only render those page which has `mathjax: true` in Front-matter.
  # If you set it to false, it will load mathjax / katex srcipt EVERY PAGE.
  per_page: true

  # hexo-renderer-pandoc (or hexo-renderer-kramed) required for full MathJax support.
  mathjax:
    enable: true
    # See: https://mhchem.github.io/MathJax-mhchem/
    cdn: //cdn.jsdelivr.net/npm/mathjax@2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML
    mhchem: false

  # hexo-renderer-markdown-it-plus (or hexo-renderer-markdown-it with markdown-it-katex plugin) required for full Katex support.
  katex:
    enable: false
    # See: https://github.com/KaTeX/KaTeX/tree/master/contrib/copy-tex
    copy_tex: false
```
设置属性`enable`为`true`，即打开数学公式渲染

属性`per_page`表示是否自动渲染每一页，如果为`true`就只渲染配置块中包含`mathjax: true`的文章

    ---
    title: Next Post
    date: 2019-01-19 17:36:13
    mathjax: true
    ---

## 附加问题
#### 一行只能渲染一个行内公式，多个公式一起就不成功了

#### `_`等符号解释为markdown符号

参考[hexo Next主题中支持latex公式(转) ](http://layty.coding.me/2018/09/21/hexo/hexo-Next%E4%B8%BB%E9%A2%98%E4%B8%AD%E6%94%AF%E6%8C%81latex%E5%85%AC%E5%BC%8F/)，渲染插件`hexo-renderer-kramed`针对行内公式渲染有语义冲突，比如对于下划线等符号会转换成`markdown`语法

进入`node_modules/kramed/lib/rules/inline.js`

修改`escape`变量

```
// escape: /^\\([\\`*{}\[\]()#$+\-.!_>])/,
escape: /^\\([`*\[\]()#$+\-.!_>])/,
```

修改`em`变量

```
// em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
em: /^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
```


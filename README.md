# Pattern formation in reaction-diffusion systems: Oscillations in time and space
## Jupyter tutorial
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/axyorah/reaction_diffusion_tutorial/master?filepath=main.ipynb)

<br>

<a name="table"></a><font size=4><b>Table of contents:</b></font>
- [Requirements](#requirements)
- [Installation](#installation)
- [Introduction: So what's it all about anyway](#introduction)
- [Links to the actual content](#links)

<br>

<a name="requirements"></a><font size=4><b>Requirements</b></font>

This tutorial is written as a collection of interactive jupyter notebooks (Python3 runtime). It uses [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [bokeh](https://bokeh.pydata.org/en/latest/), [tqdm](https://pypi.org/project/tqdm/), [pillow](https://pillow.readthedocs.io/en/5.3.x/) and [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/).

If you're using pip:
```
pip install numpy
pip install scipy
pip install bokeh
pip install tqdm
pip install pillow
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

<br>

<a name="installation"></a><font size=4><b>Installation</b></font>

- Cloning from github:

```
git clone https://github.com/axyorah/reaction_diffusion_tutorial
```

- If you just want to check the static files in your browser but github doesn't render jupyter notebooks correctly (this happens often apparently) - follow this link to nbviewer: 

>https://nbviewer.jupyter.org/github/axyorah/reaction_diffusion_tutorial/blob/master/main.ipynb


- If you want to **interact** with the files in your browser (and you should) - follow this link to binder: 

>https://mybinder.org/v2/gh/axyorah/reaction_diffusion_tutorial/master?filepath=main.ipynb

<br>

<a name="introduction"></a><font size=4><b>So what's it all about anyway</b></font>

Oscillating dynamic systems are fascinating! And omnipresent... No matter your background, you've probably come across some oscillators during your studies, be it in a form of an [RLC contour](https://en.wikipedia.org/wiki/RLC_circuit) in your electrical engineering class or in a form of a [circadian clock](https://en.wikipedia.org/wiki/Circadian_clock) in your biochemistry class.

<table>
<td>
<img width="300" height="200" src="images/briggs-rauscher.gif"></img>
<body><center><font size="2"><br><a href="https://en.wikipedia.org/wiki/Briggs%E2%80%93Rauscher_reaction">Briggs–Rauscher</a> reaction  is a well-known chemical oscillator (<a href="https://www.youtube.com/watch?v=WpBwlSn1XPQ">vid</a>)</font></center></body>
</td>
</table>

Oscillations don't always come in the same flavour. Briggs-Rauscher reaction in the video above is definitely an example of an oscillating system. But so are the dots on a fur of a cheetah and patches of grass in an arid dryland! Although, if the Briggs-Rauscher reaction is an example of a system, that is oscillatory in time and is perfectly stable (homogenous) in space, the patterns on animal fur and the patches of grass, quite the opposite, are the examples of systems, that are oscillatory in space and perfectly stable in time. The idea that initially homogenous systems can spontaneously destabilize and develop spatial patterns was discussed back in the 1952 by Alan Turing in [Chemical Basis of Morphogenesis](http://www.dna.caltech.edu/courses/cs191/paperscs191/turing.pdf). This seminal paper showed that the formation of many complex spatial patterns can be nicely described by all too familiar maths (and physics)! Which is really good news for anyone who would want to recreate the formation of such patterns _in silico_.

<table>
<td>
<img width="250" height="150" src='images/Cheetah_pattern.jpg'></img>
<body><center><br>Stationary dotted pattern of cheetah (<a href="https://upload.wikimedia.org/wikipedia/commons/6/68/Cheetah_%28Kruger_National_Park%2C_South_Africa%2C_2001%29.jpg">img</a>)</center></body>
<td>
<td>
<img width="213" height="150" src='images/grass_patches_dryland.png'></img>
<body><center><br>Stationary labirynth-like grass pattern of drylands (<a href="https://www.mmnp-journal.org/articles/mmnp/pdf/2011/01/mmnp20106p163.pdf">by E. Meron</a>)</center></body>
</td>
</table>

Now you might be thinking: ok, I can picture systems that are periodic in time and stable (homogenous) in space - they behave like chemical oscillators in a well-stirred beaker; I can also picture systems that are periodic in space and stable in time - these are the systems, that, e.g., describe the formation of patterns on animal fur. But what about the systems that oscillate in both time and space? That's a great thing to wonder about, as diversity of patterns and complexity of behaviours, emerging in such systems, is mind-boggling! In literature you would usually find mentions of spirals, standing waves and traveling waves, which often emerge in such systems. In physical world these patterns occur, e.g., in a petri dish with [Belousov-Zhabotinsky](https://en.wikipedia.org/wiki/Belousov%E2%80%93Zhabotinsky_reaction) reaction. But there's so much more to it! In autocatacytic Reaction-Diffusion system described by the Gray-Scott model you can stumble upon dynamic patterns that resemble cell division, coral growth and even the formation of exotic [U-skates](http://mrob.com/pub/comp/xmorphia/uskate-world.html)! Robert Munafo's [xmorphia](http://mrob.com/pub/comp/xmorphia/index.html) is a great resource to explore all the bizzare patterns that emerge in Gray-Scott system. Here are some more examples:

<table>
<td>
    <img src="images/belousov-zhabotinsky.gif" style="width:320px;height:200px;"></img>
    <body><center> <br>Traveling waves in <br> <a href="https://en.wikipedia.org/wiki/Belousov%E2%80%93Zhabotinsky_reaction">Belousov-Zhabotinsky</a> reaction (<a href="https://www.youtube.com/watch?v=PpyKSRo8Iec">vid</a>)</center> </body>
<td>    

<td>
<img src='images/gray-scott-corals.gif' style="width:250px;height:200px;"></img>
<body><center><br>"Coral growth" <br>in Gray-Scott Reaction-Diffusion system (<a href="http://www.karlsims.com/rd.html">vid</a>)</center></body>
<td>
    
<td>
    <img src='images/Gray_Scott_F620_k609_fr1248.gif' style="width:250px;height:200px;"></img>    
    <body><center> <br>"U-skates" emerging <br>in Gray-Scott Reaction-Diffusion system (<a href="http://mrob.com/pub/comp/xmorphia/index.html#formula">vid</a>)</center></body>
<td>
</table>

In this notebook we will NOT explore all the complexity of the emergent behaviour in Reaction-Diffusion systems. Instead we will learn how to set up such systems from scratch in python and how to systematically search for interesting patterns. I will just go ahead and assume that most people reading this are curious about the topic, but hadn't had a proper time to explore it yet. So you probably still remember some bits and pieces from Calculus and Linear Algebra, but wouldn't mind a recap. Of course, if you feel you're well versed in both, feel free to skip the recap-sections. 

<br>

<a name="links"></a><font size=4><b>Links to the actual content</b></font>

The whole notebook is divided into two parts:
- [**Translating the maths behind the reaction-diffusion system into a code**](pde2code.ipynb). In this part we'll write a code for generic reaction-diffusion system, which you should be able to easily customize later. We'll brush over some concepts from calculus, but in general this part will be relatvely low on maths. 

- [**Figuring out what makes system behave in an _interesting_ way**](parameters2behaviour.ipynb). Here we'll work with much simpler systems compared to the previous part, but we'll be studying them more systematically. At the end you should have an idea of which settings to toggle to make the system behave in a way you want to... more or less... This part has linear algebra galore! 

><font size=2>
  In both parts we'll be working with _two_ system representations: "physical" and <a href="https://en.wikipedia.org/wiki/State-space_representation">state-space</a>. Physical representation uses notation familiar from textbooks, so we'll use this representation when initially working out the equations (e.g., we'll use <i><b>c</b></i> for concentrations/expression levels and <i><b>k</b></i> for reaction rate coefficients). State-space representation abstracts away from any physical meaning, which obfuscates things a little, but at the same time makes it more convenient to use general-purpose math tools. We'll mostly use state space notation in the code (e.g., we'll use <i><b>x</b></i> to denote system state variables, <i><b>u</b></i> to denote external inputs and <i><b>p</b></i> to denote system parameters)</font>. 

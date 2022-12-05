---
categories:
- programming
date: '2020-12-22T05:50:36+11:00'
image: /images/open_source_licenses.png
title: Open Source Licenses for Data Processing Code
---

When a program primarily sources and transforms data then copyleft licenses add very little protection over other open source licenses.
Because of this I've licensed my open data processing code as MIT because more complex licenses would prevent other people from using it, without adding much sharing.

There are three main license types that are used in Open Source; [MIT, Apache and GPL](https://exygy.com/blog/which-license-should-i-use-mit-vs-apache-vs-gpl/) (with [BSD family](https://en.wikipedia.org/wiki/BSD_licenses) somewhere between MIT and Apache).
The main benefit of GPL licenses is that if someone modifies and redistributes the code then they have to make the source code available, which fosters sharing and gives more rights to the users of the modified code.
However data processing code typically sits on a server somewhere and is never distributed; it's either used as a service or to build data products.
Even the most viral of the GPL licenses, the [AGPL](https://www.gnu.org/licenses/agpl-3.0.en.html), only requires providing source when the copy is *conveyed* (transferred) to someone.
Even MongoDBs anti-cloud provider extension of the AGPL, the [Server Side Public License (SSPL)](https://www.mongodb.com/licensing/server-side-public-license), only applies in the service use.
If someone wants to build their own data assets using the data processing code none of these licenses require them to share their modifications.

The copyleft licenses have a chilling effect on adoption compared with the broader licenses.
People building commercial products often get scared of GPL licenses (and especially the AGPL and SSPL), in large part because they are largely untested in court and so there is legal risk in how they would be interpreted.
This means using a GPL style license means fewer people in industry will use the code, even if a lay reading of the license says there is no restriction.

I'm mainly writing data processing code to build a portfolio, and find some data driven insights.
I want people to look at the code, see that it works and use it to build their own insights (even if it's proprietary).
If I find a commercialisation opportunity then I would keep that part closed source, but while I'm just exploring I'd rather share my findings on this website and provide people the means to follow along.
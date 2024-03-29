---
categories:
- general
date: '2020-07-18T22:07:49+10:00'
image: /images/viewed_chrome.png
title: Embrace, Extend and Extinguish
---

In the 90s Microsoft famously used a strategy of embracing other protocols, then adding extensions to their implementation until it's no longer compatible and utilising their market leverage to extinguish competing implementations.
While "EEE" is normally associated with Microsoft many of the software titans use it as an effective strategy to further their existing dominance into new markets.

Embracing a technology with an existing market is an effective way to quickly gain adoption.
There are huge numbers of people sending and receiving emails with SMTP/POP/IMAP, distributing and receiving podcasts via RSS, and publishing and viewing websites via HTML/CSS/Javascript.
If you want to break into one of these markets with both producers and consumers starting with an existing technology allows you to acquire both sides of the market.

Extending a technology on a shared platform helps the market leader lock in existing customers.
Adding extensions that are difficult to implement for technical, social or legal reasons means that it's harder for existing users to migrate away from the implementation.
When users start requiring these extensions it means that other users must migrate to the extended implementation if they want to continue to communicate.

Embracing a technology allows breaking into an existing market, and when there is enough market share extending drives more users towards the platform through network effects.
If the dominance and network effects are strong enough then the competitors will effectively be extinguished.
Unless they extend away there's no barrier from migrating from competitors.

Microsoft is well known for these techniques for web technologies.
They adopted open web standards in Internet Explorer allowing people to access the internet.
At the same time they introduced proprietary extensions such as ActiveX, which allowed websites to provide extra functionality specifically to Windows users on Internet Explorer.
Because Windows had such market dominance many developers were willing to build these extensions, which meant consumers were pushed to use IE in Windows to access the content, increasing market dominance and acceptance of the extensions.
While Microsoft didn't win the Internet, the same opportunity can be seen now with Google Chrome which has dominance of the browser market (outside of the substantial iOS market); all that's needed are compelling extensions the competitors can't emulate.

Amazon Web Services has its own related mechanism of devouring open source served products.
AWS pioneered a very effective method of billing for access to infrastructure and managed services.
Because services are billed in small increments using a new service does not trigger a procurement process.
The micro purchasing decisions end up in the hands of developers, and they actively educate them on best practices that involve a large number of services, which makes products hard to migrate.
They also have minimal charges for data ingress, but hefty fees for data egress making it cheap to migrate to but expensive to offload processing to other cloud providers.

AWS embraces open source server technologies, building their own managed services of open source technologies such as MongoDB, Elasticsearch, Redis, MySQL, Postgres, Presto and Kubernetes.
This means many developers will use the AWS provided solution rather than manage their own MongoDB instance, and can easily migrate an existing self-hosted service.
However they also extend their offerings to work better with other Amazon managed services, further driving developers against self hosting.
Many of these platforms monetise by providing managed services and AWS is extinguishing them; MongoDB and Redis have changed their licence to try to block this, but I don't think it has been a large blocker because AWS is so much better resources than these competitors.

Another example is Apple iMessage.
The standards of SMS and MMS are available on Apple phones allowing them to communicate with other forms via this standard mechanism.
However they extend this between iOS devices to have extra features and a differentiated experience, making it more appealing to interact with other iOS users.
Because Apple is the market leader in mobile it pushes people towards iOS, and it makes it harder for people to switch away from iOS; they end up losing their history and missed messages in the transition until they can work out how to switch it off.

It's not just for software giants either.
The messaging client Slack originally embraced other chat protocols like IRC by providing gateways/bridges.
They would enable you to send and receive messages between Slack and IRC.
This meant they could get users using other chat clients to communicate on their platform without the friction of forcing them to switch.
At the same time they built many extensions that weren't available on existing protocols, and so encouraged migration to Slack itself.
As it gets to a critical mass Slack is slowly removing the ability to create these extensions by [removing test tokens](https://api.slack.com/changelog/2020-02-legacy-test-token-creation-to-retire) and pushing them to more limited Slack apps, moving into the Extinguish phase.

Embrace and extend is an effective strategy for technology companies with leverage to dominate a new market.
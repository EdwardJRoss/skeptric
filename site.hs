--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
import           Data.Monoid (mappend)
import           Hakyll
import           Text.Pandoc.Options


--------------------------------------------------------------------------------
main :: IO ()
main = hakyllWith config $ do
    match "images/*" $ do
        route   idRoute
        compile copyFileCompiler

    match "favicon.ico" $ do
      route idRoute
      compile copyFileCompiler

    match "css/*" $ do
        route   idRoute
        compile compressCssCompiler

    match (fromList ["about.md"]) $ do
        route   $ setExtension "html"
        compile $ pandocCompiler
            >>= loadAndApplyTemplate "templates/default.html" defaultContext
            >>= relativizeUrls

    match "posts/*" $ do
        route $ setExtension "html"
        compile $ compiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx
            >>= relativizeUrls

    match "index.html" $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let indexCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    defaultContext

            getResourceBody
                >>= applyAsTemplate indexCtx
                >>= loadAndApplyTemplate "templates/default.html" indexCtx
                >>= relativizeUrls

    match "templates/*" $ compile templateCompiler

config = defaultConfiguration 
  {
    deployCommand = "rsync -rvv _site/ edwardro@edwardross.id.au:public_html/"
  }

--------------------------------------------------------------------------------
pandocReaderOptions :: ReaderOptions
pandocReaderOptions = def
  {
    readerSmart = True,
    readerApplyMacros = True
  }

pandocWriterOptions :: WriterOptions
pandocWriterOptions = def
  {
    writerHTMLMathMethod = MathML Nothing
  }
compiler = pandocCompilerWith pandocReaderOptions pandocWriterOptions
--------------------------------------------------------------------------------
postCtx :: Context String
postCtx =
    dateField "date" "%B %e, %Y" `mappend`
    dateField "fulldate" "%FT%T%z" `mappend`
    defaultContext

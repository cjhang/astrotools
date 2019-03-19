#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class SpecToolsError(Exception):
    ''' AstroTools Error handler'''
    def __init__(self, message=None):
        
        message = 'Unknown SpecTools Error' if not message else message

        # add contact imformation
        giturl = 'https://github.com/cjhang/spectools/issues/new'
        self.message = message + '\nIssues tracking: \n ' + giturl \
                  + '\nPlease Fill out necessary describing information,' \
                  + 'if possible, appending full traceback information.'

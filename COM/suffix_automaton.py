# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:32:25 2019

Classes and methods for suffix automata

Details in Marceau C. "Characterizing the Behavior of a Program Using 
Multiple-Length N-grams"

@author: tangk
"""

from suffix_trees.STree import STree, _SNode
import re
from itertools import chain


class SuffixAutomaton(STree):
    def __init__(self, S, N):
        # S: string or list of strings
        # N: N gram
        self.S = S
        self.N = N
        
        
    def _create_suffix_tree(self):
        # creates the suffix tree 
        super().__init__(input=self.S)
        
        # remove all leaves ending in the terminal symbol
        U = [leaf.parent for leaf in self.root._get_leaves() if not all(map(lambda x: x.isalpha(),list(zip(*leaf.parent.transition_links))[1]))]
        for u in U:
            if len(u.transition_links)==0:
                continue
            suffix = [s for s in list(zip(*u.transition_links))[1] if not s.isalpha()]
            for s in suffix:
                self._delete_edge(u, u._get_transition_link(s))
            u_actual = u
            while len(u_actual.transition_links)==1:
                if u_actual.transition_links[0][0].is_leaf():
                    u_actual.depth = u_actual.transition_links[0][0].depth
                    self._delete_edge(u_actual, u_actual.transition_links[0][0])
                else:
                    s = self._delete_source_link(u_actual.parent, u_actual, return_suffix=True)
                    u_actual.parent._add_transition_link(u_actual.transition_links[0][0], s)
                    self._delete_source_link(u_actual, u_actual.transition_links[0][0])
                u_actual = u.parent

    def _truncate_suffix_tree(self):
        paths = {leaf.idx: leaf for leaf in self.root._get_leaves()}
        for path, l in paths.items():            
            if l.depth-1 <= self.N:
                l.depth = l.depth-1
            elif l.parent.depth < self.N:
                self._truncate_leaf(l)
            else:
                while l.parent.depth >= self.N:
                    # delete the edge leading from the parent to the leaf
                    # the parent becomes the new leaf
                    p = l.parent
                    self._delete_edge(p, l)                
                    l = p
            

    def _equivalence_compress(self):
        end_leaves = [leaf for leaf in self.root._get_leaves()]
        while len(end_leaves)>0:
            leaf = end_leaves.pop()
            s_end = self._edgeLabel(leaf, self.root)
            for i in range(2, len(s_end)+1):
                s2 = s_end[:i]
                s1 = s2[1:]
                print('Full string {0}. Trying s1={1}, s2={2}'.format(s_end, s1,s2))
                
                s2_suffixes = self._find_all_suffixes(s2, depth=self.N-len(s2))
                s1_suffixes = self._find_all_suffixes(s1, depth=self.N-len(s2))
                
                if s1_suffixes==s2_suffixes:
                    print('Suffixes equivalent!')
                    print(s1_suffixes)
                    s1_node, s1_str = self.get_node(s1)
                    s2_node, s2_str = self.get_node(s2)
                    
                    # check the nodes
                    if not s1_str.startswith(s1):
                        raise Exception('Error in finding node for s1')
                    if not s2_str.startswith(s2):
                        raise Exception('Error in finding node for s2')
                    
                    s2_leaves = s2_node._get_leaves()
                    
                    print('Leaves to delete: ', [self._edgeLabel(i, self.root) for i in s2_leaves])
                    
                    s2_node.parent._add_transition_link(s1_node, suffix=s_end[s2_node.parent.depth])
                    self._add_node_connection(s2_node.parent, s1_node)
                    for l in s2_leaves:
                        print('Deleting ' + self._edgeLabel(l, self.root))
                        l_actual = l
                        l_parent = l_actual.parent
                        while l_actual != s2_node:
                            self._delete_edge(l_parent, l_actual)
                            l_actual = l_parent
                        if l in end_leaves:
                            end_leaves.remove(l)
                    break

           
    def _find_all_suffixes(self, s, depth=1):
        # find all suffixes of the string s
        # set a depth
        r = re.compile('^' + s)
        j = len(s)
        k = len(s)+depth
        
        start_node, start_str = self.get_node(s)
        if len(start_str)>=k:
            return {start_str[j:k]}
        
        all_nodes = self._get_all_node_children(start_node, max_depth=depth)
        all_strings = [start_str + n[1] for m in all_nodes for n in m]
        
        matches = filter(r.match, all_strings)
        return set([i[j:k] for i in matches if len(i)>=k])
    
    def _add_node_connection(self, u, v):
        # add node u to the list of connections in v
        if hasattr(v, 'connections'):
            v.connections.append(u)
        else:
            v.connections = [u]
    
    def get_node(self, s, start=None):
        # follows the tree along the string s optionally from the starting node
        if start is None:
            node = self.root
        else:
            node = start
        left_s = ''
        right_s = s
        
        while len(left_s) < len(s):
            for i in right_s:
                # try each letter 
                if node._get_transition_link(i):
                    test_node = node._get_transition_link(i)
                    test_s = self._edgeLabel(test_node, node)
                    j = min(len(test_s), len(right_s))
                    if right_s[:j]==test_s[:j]:
                        left_s = left_s + test_s
                        right_s = right_s[j:]
                        node = test_node
                        break
                raise Exception('Error finding node for string ' + s)
        return node, left_s
    
    def _edgeLabel(self, node, parent):
        if node.idx==parent.idx:
            return super()._edgeLabel(node, parent)
    
    def _delete_edge(self, u, v):
        # delete link leading from u to v
        # delete u from the parent of v or from the list of connections
        self._delete_source_link(u, v)
        self._delete_target_link(u, v)
            
        # if v has no parent or transition links, redirect the connections
        if hasattr(v, 'connections') and v.parent is None and len(v.transition_links)==0:
            while len(v.connections)>0:
                w = v.connections.pop()
                suffix = self._delete_source_link(w, v, return_suffix=True)
                if v._suffix_link:
                    w._add_transition_link(v._suffix_link, suffix=suffix)
                else:
                    raise Exception('Redirect error!')
    
    
    def _delete_source_link(self, u, v, return_suffix=False):
        # delete transition link from u to v
        # optionally returns the suffix
        transition_links = u.transition_links
        link = [node for node in u.transition_links if node[0]==v]
        if len(link)==0:
            raise Exception('Error with edge deletion: no link from u to v!')
        u.transition_links.remove(link[0])
        if return_suffix:
            return link[0][1]
        
    
    def _delete_target_link(self, u, v):
        # delete the references to u in v
        if v.parent==u:
            v.parent = None
        elif hasattr(v, 'connections') and u in v.connections:
            v.connections.remove(v)
        if v._suffix_link==u:
            v._suffix_link = None
            
    
    def _truncate_leaf(self, v):
        # truncate the link leading from the root to depth N
        v.depth = self.N
    
    
    def _get_all_node_children(self, u, max_depth=1):
        # gets the children of a node u of depth up to and including max_depth
        # returns a nested list of children with increasing depth
        all_children = [[(u, '')]]
        while len(all_children)-1 < max_depth:
            nodes = all_children[-1]
            children = list(chain(*map(self._get_node_children, nodes)))
#            children = list(chain(*map(self._get_node_children, nodes)))
            all_children.append(children)
        return all_children

   
    def _get_node_children(self, node):
        # gets all the children of a node
        if node==[]:
            return None
        return [(i[0], node[1] + self._edgeLabel(i[0], node[0])) for i in node[0].transition_links]

st = SuffixAutomaton('abccabbccaabc', 4)
st._create_suffix_tree()
st._truncate_suffix_tree() 
#st._get_all_node_children(st.root, max_depth=4)
st._equivalence_compress()
